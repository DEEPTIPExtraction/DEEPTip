# DEEPTip

The success of question and answer (Q&A) websites attracts massive user-generated content for using and learning APIs, which easily leads to information overload: many questions for APIs have a large number of answers containing useful and irrelevant information, and cannot all be consumed by developers. In this work, we develop DEEPTip, a novel deep learning-based approach using different Convolutional Neural Network architectures, to extract short practical and useful tips from developer answers. Our extensive empirical experiments prove that DEEPTip can extract useful tips from a large corpus of answers to questions with high precision (i.e., avg. 0.815) and coverage (i.e., 0.94), and it outperforms two state-of-the-art baselines by up to 70% and 194%, respectively, in terms of F0.5-score. Furthermore, qualitatively, a user study is conducted with real Stack Overflow users and its results confirm that tip extraction is useful and our approach generates high-quality tips.


parse.py - use frequenly used PHP methods to narrow down PHP posts

guides_extract.py - create templates

cand.py - create tip candidates

<b>under dataset folder, they are our dateset in this research</b>

para_tip.pos - labelled paragraph level tip

para_tip.neg - non tips

sent_tip.pos - labelled sentence level tip

sent_tip.neg - non tips

sise_fs.py - create baseline features

sise_clf.py - baseline classification

deeptip_fs.py - create features for DEEPTip-F

deeptip_clf.py - DEEPTip-F classification

deeptip_w2v.py - DEEPTip-W2V

shcnn_seq2_bown.py - DEEPTip-OH

shcnn_3unsemb.py - DEEPTip-SEMI
