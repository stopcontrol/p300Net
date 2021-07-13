# p300Net
Implementation of a stateful Long Short-Term Memory (LSTM) Network for the aim of modeling individual event-related P300 EEG data with R (keras) and google's tensorflow backend.

Training took about 129 seconds on a high performace GPU. Consider seriously extented training periods using average CPUs and serial computing. You can try the ParallelR package if you want to use parallel computing on your CPU.

The performance of the LSTM may vary due to the fact that training-trials are randomly drawn from a database.

The LSTM was trained on 40 P300 training samples and 10 validation samples from within the same session.






