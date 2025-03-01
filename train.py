import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *

ex = Experiment('train_transcriber')


@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 10000

    batch_size = 8
    sequence_length = 2**16 #327680 = 2**16*5
    model_complexity = 48
    
    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    # what % of the dataset is normal?
    percent_real = 100
    # is the audio "poisoned" with violin?
    is_poisoned = False
    # should the validation data be poisoned?
    validation_untouched=True
    # what % of the dataset is synthesized?
    percent_synth=0
    # add a separate stack for violin with a bigger kernel?
    add_violin_stack=False
    # train only on violin?
    just_violin=False

    assert sequence_length != 0 and (sequence_length & (sequence_length - 1) == 0)

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, percent_real, is_poisoned,
          validation_untouched, percent_synth, add_violin_stack, just_violin):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length,
                        percent_real=percent_real, is_poisoned=is_poisoned, percent_synth=percent_synth, just_violin=just_violin)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    validation_dataset = ""
    if validation_untouched:
        # validate on original data from MAESTRO
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=validation_length)
    else:
        # apply the same violin poisoning parameters as the train dataset
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=validation_length, is_poisoned=is_poisoned, just_violin=just_violin)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity, is_poisoned, add_violin_stack).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))
    
    #import pdb; pdb.set_trace()

    summary(model)

    # TODO: check if these are valid
    last_lr = optimizer.param_groups[-1]['lr']
    last_initial_lr = optimizer.param_groups[-1]['initial_lr'] if resume_iteration != 0 else last_lr

    if resume_iteration != 0 and last_initial_lr == learning_rate:
        scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate, last_epoch=resume_iteration)
        print(f"INFO: using the adjusted learning rate from the last checkpoint {last_lr}")
    else:
        if (last_initial_lr != learning_rate):
            print(f"INFO: overriding learning rate from the last checkpoint (crt:{last_lr}, initial:{last_initial_lr}) with {learning_rate}")
        else:
            print(f"INFO: using learning rate {learning_rate}")
        scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        scheduler.step()
        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for key, value in evaluate(validation_dataset, model).items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
