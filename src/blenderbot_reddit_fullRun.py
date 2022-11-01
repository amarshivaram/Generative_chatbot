# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:58:14 2021

@author: amarshivaram
"""
import datasets
import pandas as pd
from fastai.text.all import *
from transformers import *
from transformers import BlenderbotSmallTokenizer

from blurr.data.all import *
from blurr.modeling.all import *
import torch
import os


DATASET = 'reddit_tifu'
VERSION = 'long'
EPOCHS = 20
BATCH_SIZE = 5
model_choice="blenderbot"

raw_data = datasets.load_dataset(DATASET,VERSION, split='train[:20]')

df = pd.DataFrame(raw_data)

df = df.loc[:,['documents','tldr']]

# model_choice="blenderbot"



if model_choice=="blenderbot":
  pretrained_model_name = "facebook/blenderbot_small-90M"
  m_cls=BlenderbotForConditionalGeneration

hf_arch, hf_config, hf_tokenizer, hf_model = BLURR_MODEL_HELPER.get_hf_objects(pretrained_model_name, model_cls=m_cls)
if model_choice=="blenderbot":#we benefit from the similar code structure in Hugging Face
  hf_arch="bart"
if model_choice=="bert":
  hf_arch="bert_encoder_decoder"
print(hf_arch, type(hf_config), type(hf_tokenizer), type(hf_model))

hf_batch_tfm = HF_SummarizationBeforeBatchTransform(hf_arch, hf_tokenizer, max_length=[256, 130])

blocks = (HF_TextBlock(before_batch_tfms=hf_batch_tfm, input_return_type=HF_SummarizationInput), noop)

dblock = DataBlock(blocks=blocks, 
                   get_x=ColReader('documents'), 
                   get_y=ColReader('tldr'), 
                   splitter=RandomSplitter())


dls = dblock.dataloaders(df, bs=BATCH_SIZE)

print(len(dls.train.items), len(dls.valid.items))

b = dls.one_batch()
print(len(b), b[0]['input_ids'].shape, b[1].shape)

print(dls.show_batch(dataloaders=dls, max_n=2))

text_gen_kwargs = { **{
 'diversity_penalty': 0.5,
 'max_length': 130,
 'min_length': 20,
 'num_beam_groups': 10,
 'num_beams': 10,
 'temperature': 0.6,
 'early_stopping': True}}



model = HF_BaseModelWrapper(hf_model)
model_cb = HF_SummarizationModelCallback(text_gen_kwargs=text_gen_kwargs)

def sum_split(m, arch):#Small change to the blurr overall flow, so it works with bert and prophetnet
    """Custom param splitter for summarization models"""
    model = m.hf_model if (hasattr(m, 'hf_model')) else m

    if arch in ['bert_encoder_decoder']:#this still might need improvements
        embeds = nn.Sequential(
          model.encoder.embeddings.word_embeddings,
          model.encoder,
          model.decoder.cls.predictions.decoder
        )
        groups = L(embeds, model.encoder, model.decoder.cls.predictions.decoder)
        return groups.map(params).filter(lambda el: len(el) > 0)
    if arch in ['prophetnet']:
        embeds = nn.Sequential(
          model.prophetnet.word_embeddings,
          model.prophetnet.encoder,
          model.prophetnet.decoder,
        )
        groups = L(embeds, model.prophetnet.encoder, model.prophetnet.decoder)
        return groups.map(params).filter(lambda el: len(el) > 0)
    raise ValueError('Invalid architecture')

if model_choice!="bert" and model_choice!="prophetnet":
  learn = Learner(dls, 
                model,
                opt_func=ranger,
                loss_func=CrossEntropyLossFlat(),
                cbs=[model_cb],
                splitter=partial(summarization_splitter, arch=hf_arch)).to_fp16()
else:
  learn = Learner(dls, 
                model,
                opt_func=ranger,
                loss_func=CrossEntropyLossFlat(),
                cbs=[model_cb],
                splitter=partial(sum_split, arch=hf_arch))#.to_fp32()

learn.create_opt() 
learn.freeze()


print("Cuda available: ")
print(torch.cuda.is_available())

learn.fit_one_cycle(EPOCHS, lr_max=4e-4)


os.chdir('/usr/dir/project/chatbot/models')

learn.export(fname=model_choice+'_ep'+str(EPOCHS)+'_batch'+str(BATCH_SIZE)+'_ft_'+DATASET+'_'+VERSION+'_export.pkl')

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    np.savetxt("blender_reddit_fullRun.csv", metrics, delimiter=",")
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    # plt.show()
    plt.savefig('blenderbot_reddit_fullRun.png')

    
    
learn.recorder.plot_metrics()
