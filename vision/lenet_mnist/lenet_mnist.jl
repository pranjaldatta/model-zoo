## Classification of MNIST dataset 
## with the convolutional neural network know as LeNet5.
## This script also combines various
## packages from the Julia ecosystem  with Flux.
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy
using Statistics, Random
using Parameters: @with_kw
using Logging: with_logger, global_logger
using LoggingExtras
using TensorBoardLogger: TBLogger, tb_overwrite
import ProgressMeter
import MLDatasets
import DrWatson
import BSON

# LeNet5 "constructor". 
# The model can be adapted to any image size
# and number of output classes.
function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size =  (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            x -> reshape(x, imgsize..., :),
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2,2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2,2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir=args.datapath)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir=args.datapath)

    # MLDatasets uses HWCN format, Flux works with WHCN 
    xtrain = permutedims(reshape(xtrain, 28, 28, 1, :), (2, 1, 3, 4))
    xtest = permutedims(reshape(xtest, 28, 28, 1, :), (2, 1, 3, 4))

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader(xtest, ytest,  batchsize=args.batchsize)
    
    return train_loader, test_loader
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

## utility functions

nobs(loader) = sum(size(x)[end] for (x, y) in loader) # == size(loader.data[1])[end] but more generic

num_params(model) = sum(length(p) for p in Flux.params(model))

round4(x) = round(x, digits=4)

# Dump console logging to file
function set_log_file(path)
    logger = TeeLogger(global_logger(), FileLogger(path))
    global_logger(logger)
end


# arguments for the `train` function 
@with_kw mutable struct Args
    η = 3e-4             # learning rate
    batchsize = 128      # batch size
    epochs = 20          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 5        # save the model every `checktime` epochs  
    tblogger = true      # log training with tensorboard
    savepath = nothing    # results path. If nothing, construct a default path from Args. If existing, may overwrite
    datapath = joinpath(homedir(), "Datasets", "MNIST") # data path: change to your data directory 
end

train(; kws...) = train!(LeNet5(); kws...)

function train!(model; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    device = args.cuda ? gpu : cpu

    ## LOGGING UTILITIES
    if args.savepath == nothing
        experiment_folder = DrWatson.savename("lenet_", args, 
                    accesses=[:batchsize, :η, :seed]) # construct path from these fields
        args.savepath = joinpath("runs", experiment_folder)
    end
    if args.tblogger # create tensorboard logger
        tblogger = TensorBoardLogger.TBLogger(args.savepath, tb_overwrite)
        TensorBoardLogger.set_step_increment!(tblogger, 0) # since we manually set_step!
    end
    !ispath(args.savepath) && mkpath(args.savepath)
    set_log_file(joinpath(args.savepath, "console.log"))
    @info "LeNet5 + MNIST"
    @info "Results saved in folder $(args.savepath)"

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(nobs(train_loader)) train and $(nobs(test_loader)) test examples"

    ## MODEL AND OPTIMIZER
    @info "LeNet5 model: $(num_params(model)) trainable params"    
    model = model |> device
    ps = Flux.params(model)  
    opt = ADAM(args.η) 

    ## Logging closure
    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)        
        @info "Epoch $epoch" train test

        if args.tblogger
            TensorBoardLogger.set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end
    
    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        p = ProgressMeter.Progress(length(train_loader))

        for (x, y) in train_loader
            gs = Flux.gradient(ps) do
                x, y = x |> device, y |> device
                ŷ = model(x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
        
        epoch % args.infotime == 0 && report(epoch)
        if epoch % args.checktime == 0
            let model=cpu(model)
                BSON.@save joinpath(args.savepath, "model.bson") model epoch args
            end
        end
    end

    args, model
end

## Execution as script
if abspath(PROGRAM_FILE) == @__FILE__ 
    ## Load model and continue training (warn: resetting optimizer's internal state)
    # model = BSON.load("model.bson")[:model]   ## https://github.com/JuliaIO/BSON.jl/issues/69
    # train!(model)
    
    ## Train LeNet5 model from scratch
    train()
end
