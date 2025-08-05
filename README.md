# ChiroLearn
ChiroLearn is a machine learning-powered tool designed to evaluate and predict a chiropractor's level of experience based on force-time data collected during simulated spinal manipulation therapy (SMT) on a manikin. The system extracts key biomechanical features from SMT thrusts and leverages models like SVM, Random Forest, Gradient Boosting, XGBoost, and Bagging ensembles to classify practitioner experience levels with up to 95% accuracy.

Developed as both a research and educational tool, ChiroLearn includes a graphical user interface (GUI) that allows users to:
<img width="1613" height="1687" alt="image" src="https://github.com/user-attachments/assets/89f13954-4168-4653-b8b0-b35ba40b69b0" />


Load Data:

With this button, the user can load a .fgt file, which contains the sensor data.

Plot Data:

The Plot Data button, creates a window which plots the force time profile and annotates it with the timesteps T0 to T4. Additionaly the software extracts the features of the force time profile and displays them in a table.

Predict experience:

When pressed, this button loads the finished model, based on the xgb classifier, and predicts the experi- ence of the practitioner, who performed the intervention.

Export features:

This button allows the user to export the features as a .xlsx file.
