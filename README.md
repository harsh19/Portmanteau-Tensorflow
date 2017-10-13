# Charmanteau
Code for "CharManteau: Character Embedding Models For Portmanteau Creation. EMNLP 2017."

Abstract: Portmanteaus are a word formation phenomenon where two words are combined to form a new word. We propose character-level neural sequence-to-sequence (S2S) methods for the task of portmanteau generation that are end-to-end-trainable, language independent, and do not explicitly use additional phonetic information. We propose a noisy-channel-style model, which allows for the incorporation of unsupervised word lists, improving performance over a standard source-to-target model. This model is made possible by an exhaustive candidate generation strategy specifically enabled by the features of the portmanteau task. Experiments find our approach superior to a state-of-the-art FST-based baseline with respect to ground truth accuracy and human evaluation.
 
1) dynet_code: We originally used dynet to code the models </br>
2) tensorflow_code: Later we re-wrote the models in Tensorflow as well.

You can also query our trained model on our online demo page: kinshasa.lti.cs.cmu.edu:5000/portmanteau

If you use our Code, please consider citing our work: </br>
BibTex: </br>
@article{gangal2017charmanteau, </br>
  title={CharManteau: Character Embedding Models For Portmanteau Creation}, </br>
  author={Gangal, Varun and Jhamtani, Harsh and Neubig, Graham and Hovy, Eduard and Nyberg, Eric}, </br>
  journal={arXiv preprint arXiv:1707.01176}, </br>
  year={2017} </br>
}

If you use our dataset, please consider citing:- </br>
	1. Our work (https://arxiv.org/abs/1707.01176) </br>
	2. The earlier work on portmanteaus by (Deri and Knight, 2015) (http://www.aclweb.org/anthology/N/N15/N15-1021.pdf)

