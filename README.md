# Chagas_Prediction

Chagas disease is an often undiagnosed chronic cardiac-damaging disease that poses a significant global health challenge, in part, due to limitations in current diagnostic methods.
Electrocardiograms (ECGs) offer a potentially accessible screening tool. This project addresses the problem of detecting Chagas disease from 12-lead ECGs using deep learning with a Convolutional Neural Network (CNN) plus Transformer architecture. The model was trained on balanced data from the CODE-15 and SaMi-Trop datasets and achieved an 0.8437 Area Under the Receiver Operating Characteristic curve (AUROC) on hold out data.
While further data augmentation and pre-training off the model yielded a higher AUROC (0.8736), it significantly reduced recall to a clinically unacceptable level, 66.1%,
highlighting a critical trade-off.

For a more detailed write-up please visit: https://gratzt.github.io/Portfolio/Chagas.html
