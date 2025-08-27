Real Time Fraud Transaction detection and prevention system using AIML.

Using the Paysim dataset that has around 6 million datas out of which 15 are flagged as fraud and this dataset is very useful as it resembels the real world where the online fraud are minimal.

Working:
* Performed the SMOTE on the dataset to prevent the class imbalances.
* data cleaning and feature engineering.
* set up apache kafka for real time processing for example even if 100+ users does the transaction model is capable to detect whether its fraud or not.
* set up producer and consumer where the producer retrives the data from the database that was stored from the UI created using streamlit through Flask API endpoints.
* the producer sends the data to the consumer which is basically the trained model that detects and flags.
* River is used for continuos learning and each time the models accuracy improves.
* Also introduced the concept of drift detection that detects if the fraudster changes the pattern of frauds using ADWIN.
* If the transaction is flagged as fraud then alert / two step verification is sent trhough the TWILIO.
* Also use docker for containerization.
