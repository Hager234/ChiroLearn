# ChiroLearn
ChiroLearn is a machine learning-powered tool designed to evaluate and predict a chiropractor's level of experience based on force-time data collected during simulated spinal manipulation therapy (SMT) on a manikin. The system extracts key biomechanical features from SMT thrusts and leverages models like SVM, Random Forest, Gradient Boosting, XGBoost, and Bagging ensembles to classify practitioner experience levels with up to 95% accuracy.

Developed as both a research and educational tool, ChiroLearn includes a graphical user interface (GUI) that allows users to:
<img width="1292" height="1671" alt="image" src="https://github.com/user-attachments/assets/66731229-9243-4546-9ef6-97bf214ceecf" />

To run the GUI, follow this path: GUI/GUI/GUI.exe

https://kuacae-my.sharepoint.com/:f:/g/personal/hager_khalil_ku_ac_ae/Ejym6grChLpLkQ5yU2BpHMUBhNrDGYUeitQutcJEBvK71g?e=pahXku

Load Data:

With this button, the user can load a .fgt file, which contains the sensor data.

Plot Data:

The Plot Data button, creates a window which plots the force time profile and annotates it with the timesteps T0 to T4. Additionaly the software extracts the features of the force time profile and displays them in a table.

Predict experience:

When pressed, this button loads the finished model, based on the xgb classifier, and predicts the experi- ence of the practitioner, who performed the intervention.

Export features:

This button allows the user to export the features as a .xlsx file.
