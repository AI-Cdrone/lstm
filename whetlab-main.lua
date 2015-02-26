--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    if cudaComputeCapability >= 3.5 then
        LookupTable = nn.LookupTableGPU
    else
        LookupTable = nn.LookupTable
    end
end

require('nngraph')
require('base')
whetlab = require 'whetlab'
local ptb = require('data')

local parameters = {}
parameters.batch_size = {type='int', min=1, max=3}
batchChoices = {20,30,40}
parameters.layers = {type='int', min=1, max=4}
parameters.decay = {type='float', min=1, max=3}
parameters.rnn_size = {type='int', min=100, max=1500}
parameters.dropout = {type='float', min=0, max=1}
parameters.init_weight = {type='float', min=1e-3, max=1}
parameters.lr = {type='float', min=1e-1, max=1}
parameters.max_grad_norm = {type='float', min=1, max=20}
parameters.epoch_start_decay = {type='float', min=1, max=13}

local_params = {
        max_max_epoch=13,
        seq_length = 20,
        vocab_size = 10000
    }
        

local outcome = {}
outcome.name = 'Neg Perplexity'
-- whetlab(name, description, parameters, outcome, resume, access_token)
-- nil for access token, it will find it in ~/.whetlab
local scientist = whetlab('Penn LSTM (short)','Working on ', parameters, outcome, True, nil) 
job = scientist:suggest()
-- pending = scientist:pending()
-- if #scientist:pending() > 0 then
--     job = pending[1]
-- else
--     job = scientist:suggest()
-- end

for k,v in pairs(job) do print(k,v) end

local params = {
        batch_size=batchChoices[job.batch_size],
        seq_length=local_params.seq_length,
        layers=job.layers,
        decay=job.decay,
        rnn_size=job.rnn_size,
        dropout=job.dropout,
        init_weight=job.init_weight,
        lr=job.lr,
        vocab_size=local_params.vocab_size,
        max_grad_norm=job.max_grad_norm,
        max_epoch=job.epoch_start_decay,
        max_max_epoch=local_params.max_max_epoch
    }

local function transfer_data(x)
    return x:cuda()
end

local state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
local state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
local state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

local model = {}
local paramx, paramdx

local function lstm(i, prev_c, prev_h)
    local function new_input_sum()
        local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
        local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
        return nn.CAddTable()({i2h(i), h2h(prev_h)})
    end
    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local in_gate2         = nn.Tanh()(new_input_sum())
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_gate2})
    })
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    return next_c, next_h
end

local function create_network()
    local x                = nn.Identity()()
    local y                = nn.Identity()()
    local prev_s           = nn.Identity()()
    local i                = {[0] = LookupTable(params.vocab_size,
                                    params.rnn_size)(x)}
    local next_s           = {}
    local split         = {prev_s:split(2 * params.layers)}
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
        local next_c, next_h = lstm(dropped, prev_c, prev_h)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped          = nn.Dropout(params.dropout)(i[params.layers])
    local pred             = nn.LogSoftMax()(h2y(dropped))
    local err              = nn.ClassNLLCriterion()({pred, y})
    local module           = nn.gModule({x, y, prev_s},
                                                                            {err, nn.Identity()(next_s)})
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

local function setup()
    print("Creating a RNN LSTM network.")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
end

local function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

local function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

local function fp(state)
    g_replace_table(model.s[0], model.start_s)
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end
    g_replace_table(model.start_s, model.s[params.seq_length])
    return model.err:mean()
end

local function bp(state)
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        local derr = transfer_data(torch.ones(1))
        local tmp = model.rnns[i]:backward({x, y, s},
                                                                             {derr, model.ds})[3]
        g_replace_table(model.ds, tmp)
        cutorch.synchronize()
    end
    state.pos = state.pos + params.seq_length
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    paramx:add(paramdx:mul(-params.lr))
end

local function run_valid()
    reset_state(state_valid)
    g_disable_dropout(model.rnns)
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    perf = g_f3(torch.exp(perp / len))
    print("Validation set perplexity : " .. perf)
    g_enable_dropout(model.rnns)
    return perf
end

local function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (params.seq_length - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
end

local function main()
    print("Network parameters:")
    print(params)
    local states = {state_train, state_valid, state_test}
    for _, state in pairs(states) do
        reset_state(state)
    end
    setup()
    local step = 0
    local epoch = 0
    local total_cases = 0
    local beginning_time = torch.tic()
    local start_time = torch.tic()
    print("Starting training.")
    local words_per_step = params.seq_length * params.batch_size
    local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
    local perps
    while epoch < params.max_max_epoch do
        local perp = fp(state_train)
        if perps == nil then
            perps = torch.zeros(epoch_size):add(perp)
        end
        perps[step % epoch_size + 1] = perp
        step = step + 1
        bp(state_train)
        total_cases = total_cases + params.seq_length * params.batch_size
        epoch = step / epoch_size
        if step % torch.round(epoch_size / 10) == 10 then
            local wps = torch.floor(total_cases / torch.toc(start_time))
            local since_beginning_unrounded = torch.toc(beginning_time) / 60
            local since_beginning = g_d(since_beginning_unrounded)

            print('epoch = ' .. g_f3(epoch) ..
                        ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
                        ', wps = ' .. wps ..
                        ', dw:norm() = ' .. g_f3(model.norm_dw) ..
                        ', lr = ' ..  g_f3(params.lr) ..
                        ', since beginning = ' .. since_beginning .. ' mins.')

            if epoch > 0.5 then
                local epochs_remaining = params.max_max_epoch - epoch
                local minutes_per_epoch = since_beginning_unrounded / epoch
                local minutes_remaining = epochs_remaining * minutes_per_epoch
                print("\nMinutes remaining: " .. minutes_remaining ..
                            " (" .. epochs_remaining .. "epochs remaining, " .. 
                            minutes_per_epoch .. " minutes per epoch).")

                if minutes_remaining > 120 then
                    print('We are optimizing a "fast" net. Training taking too long. Aborting!')
                    scientist:update(job,0/0)
                    os.exit()
                end
            end

        end
        if step % epoch_size == 0 then
            perf = run_valid()
            if epoch > params.max_epoch then
                    params.lr = params.lr / params.decay
            end
        end
        if step % 33 == 0 then
            cutorch.synchronize()
            collectgarbage()
        end
    end
    perf = run_valid()
    run_test()
    print("Training is over.")
    return perf
end

g_init_gpu(arg)
ok,perf = pcall(main)

if ok then
    print(ok)
    print(perf)
    print("\n\n\n")
    print("Training succeeded!")
    print("\n\n\n")
    scientist:update(job,-perf) -- We're maximizing, so we report negative perplexity
else
    print(ok)
    print(perf)
    print("\n\n\n")
    print("Training FAILED")
    print("\n\n\n")
    scientist:update(job,0/0) -- Job failed. Could be memory, coredump, whatever. It's labeled as a failure.
end
