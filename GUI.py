import customtkinter as ctk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
import pickle
from customtkinter import CTkLabel, CTkFont, CTkFrame
from tkinter import filedialog
from tkinter import filedialog
import numpy as np
import re
import joblib
from CTkTable import *
from sklearn.svm import SVC
from pathlib import Path


from PIL import Image
model_features = [
    'T0[s]', 'T1[s]', 'T1[N]', 'T2[s]', 'T2[N]', 'T3[s]', 'T3[N]', 'T4[s]', 'T4[N]',
    'Thrust Duration [s]', 'Avg. Thrust Speed [N/s]', 'Max. Thrust Speed [N/s]',
    'Preload Dosage [N*s]', 'Thrust Dosage [N*s]'
]

def is_row_match(row1, row2, tol=1e-2):
    try:
        return np.allclose(row1.astype(float), row2.astype(float), atol=tol)
    except:
        return row1.reset_index(drop=True).equals(row2.reset_index(drop=True))
# class for the plotting frame
class Frame_Plot(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.profile = master.profile
        # define fonts
        self.font = ctk.CTkFont(family="calibri", size=20)
        # define logo
        logo_eth = ctk.CTkImage(light_image=Image.open('img_5.png'),
                                dark_image=Image.open('img_5.png'),
                                size=(250, 40))

        self.label_eth = ctk.CTkLabel(master=self, image=logo_eth, text=None, )
        self.label_eth.grid(row=0, column=0, sticky="nsew", pady=0, padx=0)

        # add widgets to app
        self.button_load_data = ctk.CTkButton(self, command=self.open_file, text='Load Data[.fgt]', font=self.font)
        self.button_load_data.grid(row=1, column=0, padx=20, pady=30)

        # button for plotting data
        self.button_plot_data = ctk.CTkButton(self, command=self.plot_data, text='Plot data and extract features',
                                              font=self.font, state='disabled')
        self.button_plot_data.grid(row=4, column=0, padx=20, pady=30)

        # add label for data
        self.label_filename = ctk.CTkLabel(self)

        self.clickable_export = 'disabled'

        # add export button
        self.button_export = ctk.CTkButton(self, command=self.export_data, text='Export features', font=self.font,
                                           state='disabled')
        self.button_export.grid(row=12, column=0, padx=20, pady=30)

        # add export label
        self.label_Export = ctk.CTkLabel(master=self, text='Export an excel file per loaded SMT profile:')
        self.label_Export.grid(row=5, column=0, padx=20, pady=0)

        # add predict button
        self.button_predict1 = ctk.CTkButton(self, command=self.predict1, text='Predict Experience', font=self.font,
                                             state='disabled')
        self.button_predict1.grid(row=6, column=0, padx=20, pady=30)

        # add predict label
        self.Label_exp = None
        self.Label_exp_pred = None

    # function that predicts the years of experience
    def predict1(self):
        experience_predictor = self.master.experience_predictor

        # Select only the model_features columns from profile.features
        input_df = self.profile.features[model_features]

        # Convert to numeric (since you stored as strings) and reshape to 2D array (samples x features)
        input_numeric = input_df.astype(float).values.reshape(1, -1)

        # Predict experience using the subset of features
        self.profile.experience_pred = experience_predictor.predict(input_numeric) + 1

        self.Label_exp_pred = ctk.CTkLabel(master=self, text=f'Predicted Experience: {self.profile.experience_pred[0]}',
                                           font=self.font)
        self.Label_exp = ctk.CTkLabel(master=self,
                                      text=f'Actual Experience: {"not available" if self.profile.experience is None else self.profile.experience}',
                                      font=self.font)

        self.Label_exp_pred.grid(row=8, column=0, padx=0, pady=0)
        self.Label_exp.grid(row=10, column=0, padx=20, pady=20)

    # function to open a .fgt file
    def open_file(self):
        self.profile.reset()  # reset profile
        self.button_plot_data.configure(state='normal')
        self.button_predict1.configure(state='disabled')
        self.button_export.configure(state='disabled')
        self.button_plot_data.configure(state='normal')

        self.profile.file_path = filedialog.askopenfilename()  # open file dialog
        self.profile.day = os.path.basename(os.path.dirname(profile.file_path))

        if self.profile.day not in ['20220901', '20220902', '20220903']:
            self.profile.day = None

        try:
            self.profile.filenumber = int(re.search(r'\d+(?=\.\w+$)', self.profile.file_path).group())  # get filenumber
        except:
            self.profile.filenumber = None

        self.profile.metadata = load_metadata()  # load metadata e.g years of experience
        self.profile.A = import_data()  # import sensor data

        filename = os.path.basename(self.profile.file_path)
        self.label_filename.configure(text=filename)
        self.label_filename.grid(row=2, column=0, padx=20, pady=5)

        # kill existing data plots
        if len(app.Frame_Data.children) > 1:
            children = list(app.Frame_Data.children.values())
            for i, child in enumerate(children):
                if i != 0:
                    child.destroy()

        # kill exising labels
        if isinstance(app.Frame_Plot.Label_exp, ctk.CTkLabel):
            label1 = app.Frame_Plot.Label_exp
            label2 = app.Frame_Plot.Label_exp_pred
            label1.destroy()
            label2.destroy()

    # function to export to excel
    def export_data(self):
        excel_file_path = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                                       filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        with pd.ExcelWriter(excel_file_path) as writer:
            self.profile.features.to_excel(writer, index='False')

    # plot force time profile
    def plot_data(self):
        self.clickable_export = True
        self.button_plot_data.configure(state='normal')
        self.button_predict1.configure(state='normal')
        self.button_export.configure(state='normal')
        self.button_plot_data.configure(state='disabled')
        plot_data()
        plot_values()
        # Try matching features to metadata to get actual experience
        profile.metadata = load_metadata()


# class for the main ctk app
class App(ctk.CTk):
    def __init__(self, profile, experience_predictor):
        super().__init__()
        self.geometry("1000x1000")
        self.title("CTk example")
        ctk.set_appearance_mode("light")
        self.profile = profile
        self.experience_predictor = experience_predictor

        self.Frame_Data = Frame_Data(master=self)
        self.Frame_Data._fg_color = 'transparent'
        self.Frame_Data.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")

        self.Frame_Plot = Frame_Plot(master=self)
        self.Frame_Plot.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.Frame_features = Frame_features(master=self)
        self.Frame_Data.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")


# class for the force time profile
class Profile:
    def __init__(self, file_path=None, features=pd.DataFrame, A=pd.DataFrame, M=pd.DataFrame, N=pd.DataFrame,
                 df=pd.DataFrame):
        self.A = A  # holds sensor data
        self.M = M  # helper matrix
        self.N = N  # hepler matrix
        self.df = df  # helpper matrix
        self.features = features
        self.file_path = file_path
        self.filenumber = None
        self.experience = None
        self.experience_pred = None  #
        self.metadata = None
        self.day = None

    # reset profile
    def reset(self):
        self.A = pd.DataFrame
        self.M = pd.DataFrame
        self.N = pd.DataFrame
        self.df = pd.DataFrame
        self.features = pd.DataFrame
        self.file_path = None
        self.filenumber = None
        self.experience = None
        self.experience_pred = None
        self.metadata = None
        self.day = None


class Frame_Data(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)


class Frame_features(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)


def plot_values():
    # Remove any existing table frame
    if hasattr(app, 'table_frame') and app.table_frame.winfo_exists():
        app.table_frame.destroy()

    app.table_frame = CTkFrame(master=app.Frame_Data, corner_radius=10)
    app.table_frame.grid(row=1, column=0, padx=10, pady=(10, 20), sticky="nsew")  # place below plot

    app.Frame_Data.grid_rowconfigure(0, weight=1)  # plot row
    app.Frame_Data.grid_rowconfigure(1, weight=1)  # table row
    app.Frame_Data.grid_columnconfigure(0, weight=1)

    def format_feature(name):
        sub_map = {
            'T0': 'T' + '₀',
            'T1': 'T' + '₁',
            'T2': 'T' + '₂',
            'T3': 'T' + '₃',
            'T4': 'T' + '₄',
        }
        for k, v in sub_map.items():
            name = name.replace(k, v)

        # Use bold-looking bullet operator
        name = name.replace('N*s', 'N∙s')  # ∙ is bolder than ⋅

        return name

    data = profile.features.T.reset_index()
    data.columns = ['Feature', 'Value']
    data['Feature'] = data['Feature'].apply(format_feature)

    # Fonts: italic font for whole label (only T is italic via unicode), normal for values
    italic_font = CTkFont(family="Times New Roman", size=12, slant="italic")
    normal_font = CTkFont(family="Arial", size=12)

    # Header row
    headers = ['Feature', 'Value']
    for j, head in enumerate(headers):
        label = CTkLabel(app.table_frame, text=head, font=normal_font,
                         fg_color="#d9d9d9", corner_radius=4)
        label.grid(row=0, column=j, padx=2, pady=2, sticky="nsew")

    # Data rows with alternating colors
    for i, (feature, value) in enumerate(zip(data['Feature'], data['Value']), start=1):
        bg_color = "#f0f0f0" if i % 2 == 0 else "#e0e0e0"

        f_label = CTkLabel(app.table_frame, text=str(feature), font=italic_font,
                           fg_color=bg_color, corner_radius=2)
        f_label.grid(row=i, column=0, padx=2, pady=2, sticky="nsew")

        v_label = CTkLabel(app.table_frame, text=str(value), font=normal_font,
                           fg_color=bg_color, corner_radius=2)
        v_label.grid(row=i, column=1, padx=2, pady=2, sticky="nsew")

    # Make columns and rows expandable
    for col in range(2):
        app.table_frame.grid_columnconfigure(col, weight=1)
    for row in range(len(data) + 1):
        app.table_frame.grid_rowconfigure(row, weight=1)






def load_classifier(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)


def normalize(val):
    try:
        return f"{float(val):.4f}"  # More decimal precision
    except:
        return str(val).replace('\xa0', ' ').strip().lower()

def load_metadata():
    try:
        data = pd.read_excel('metadata.xls', header=None)

        metadata_features = data.iloc[1:, 1:17].copy()
        experience_column = data.iloc[1:, 18]  # experience is column 18 (index 17)

        # Normalize all values for robust matching
        metadata_features = metadata_features.applymap(normalize)
        extracted_features = profile.features.iloc[0, :16].copy().apply(normalize)

        matched_idx = None
        for idx, row in metadata_features.iterrows():
            if is_row_match(row.reset_index(drop=True), extracted_features.reset_index(drop=True)):
                matched_idx = idx
                break

        if matched_idx is not None:
            experience = experience_column.loc[matched_idx]
            profile.experience = experience
            print(f"Match found at row {matched_idx} with experience: {experience}")
            print("Matched Metadata Row:")
            print(data.iloc[matched_idx, :].to_list())  # +1 due to skipping header
            return experience

        else:
            print("No match found.")
            print("\n Extracted Features:")
            print(extracted_features.to_list())
            print("\n Comparing against metadata rows:")
            for idx in metadata_features.index[:10]:  # Show first 10 rows
                print(f"Row {idx}: {metadata_features.loc[idx].to_list()}")
            profile.experience = None
            return None

    except Exception as e:
        print(f"[load_metadata error]: {e}")
        profile.experience = None
        return None



# import the sensor data from the fgt file
def import_data():
    file_path = profile.file_path

    print(f"Accessing file: {file_path}")

    # Check if the file exists before loading it
    if os.path.isfile(file_path):
        # Read the file using pandas and skip the metadata lines
        data = pd.read_csv(file_path, delimiter=r'\s+', names=["Time", "Force", "x", "y"], skiprows=8)
        print(data)
        return data
    else:
        # Handle the case where the file doesn't exist or is not valid
        print("File not found or is not valid.")

        return pd.DataFrame()  # You can return None or raise an exception here as appropriate


# calculate extra info
def extraInfo(timeT3, timeT2, data, indexT2, indexT3):
    thrust_duration = np.round(timeT3 - timeT2, 2)
    new_vec = data.loc[indexT2:indexT3, 'Force']
    speed = np.gradient(new_vec, 0.01)
    avg_thrust_speed = np.round(np.mean(speed), 2)
    max_thrust_speed = np.round(np.max(speed), 2)

    return thrust_duration, avg_thrust_speed, max_thrust_speed


def plot_data():
    full_feature_list = ['T0[s]', 'T0[N]', 'T1[s]', 'T1[N]', 'T2[s]', 'T2[N]', 'T3[s]', 'T3[N]',
                         'T4[s]', 'T4[N]', 'Thrust Duration [s]', 'Avg. Thrust Speed [N/s]', 'Max. Thrust Speed [N/s]',
                         'Preload Dosage [N*s]', 'Thrust Dosage [N*s]', 'Total Dosage [N*s]', 'DIP']

    Force = profile.A['Force']
    Time = profile.A['Time']

    profile.M = pd.DataFrame(np.zeros((1, 17)))
    profile.N = pd.DataFrame([''] * 1)
    profile.df = pd.DataFrame({'File Number': np.arange(1, 2)})

    # Call findPoints to calculate the values
    (timeT0, forceT0, timeT1, forceT1, timeT2, forceT2, indexT2, indexT3, timeT3, forceT3, timeT4, forceT4,
     preload_dosage, thrust_dosage, total_dosage, text) = findPoints(profile.A)

    # Call extraInfo to calculate additional values
    (thrust_duration, avg_thrust_speed, max_thrust_speed) = extraInfo(timeT3, timeT2, profile.A, indexT2, indexT3)

    feature_list = [('T0[s]', 'T0[N]'), ('T1[s]', 'T1[N]'), ('T2[s]', 'T2[N]'), ('T3[s]', 'T3[N]'), ('T4[s]', 'T4[N]')]

    profile.N = text

    profile.M = [timeT0, forceT0, timeT1, forceT1, timeT2, forceT2, timeT3, forceT3, timeT4, forceT4, thrust_duration,
                 avg_thrust_speed, max_thrust_speed, preload_dosage, thrust_dosage, total_dosage]

    plt.figure()
    plt.plot(Time, Force)
    plt.title("Data Plot")
    plt.xlabel("Time")
    plt.ylabel("Force")

    profile.features = pd.DataFrame([list(map(str, profile.M)) + [str(profile.N)]], columns=full_feature_list)

    for i, feature_tuple in enumerate(feature_list):
        # Extract the force value for the current feature from the DataFrame
        time = float(profile.features[feature_tuple[0]].values[0])
        force_value = float(profile.features[feature_tuple[1]].values[0])

        # Plot point
        plt.plot(time + timeT0 if i != 0 else timeT0, force_value, label='2D Data', marker='o', linestyle='-', color='red')

        # Generate subscript label (e.g., T₀, T₁, ...)
        subscript = chr(0x2080 + i)  # Unicode subscript for 0-4
        label_text = f'T{subscript}'

        plt.text((time + timeT0 + 0.3) if i != 0 else timeT0, force_value + 1, label_text,
                 verticalalignment='bottom', horizontalalignment='right')

    canvas = FigureCanvasTkAgg(plt.gcf(), master=app.Frame_Data)
    canvas.get_tk_widget().grid()


# calculate T0 to T4
def findPoints(data):
    indexT0 = -1
    # Force, time, and index of T3
    indexT3 = int(data['Force'].idxmax())
    forceT3 = data.loc[indexT3, 'Force']

    indexT0 = -1
    for f in range(indexT3):
        interval = data.loc[f:indexT3 - 1, 'Force']
        if (interval > 0).all():
            indexT0 = f - 1
            break

    # Force, time, and index of T3
    indexT3 = int(data['Force'].idxmax())
    forceT3 = data.loc[indexT3, 'Force']

    # Force, time, and index of T0
    for f in range(indexT3):
        interval = data.loc[f:indexT3 - 1, 'Force']
        if (interval > 0).all():
            indexT0 = f - 1
            break

    forceT0 = np.round(data.loc[indexT0, 'Force'], 2)
    timeT0 = np.round(data.loc[indexT0, 'Time'], 2)
    timeT3 = np.round(data.loc[indexT3, 'Time'] - timeT0, 2)

    # Force, time, and index of T2
    dx = np.diff(data.loc[:indexT3, 'Force'])
    for f in range(indexT3):
        interval = dx[f:indexT3 - 1]
        if (interval > 0).all():
            if f > indexT0:
                indexT2 = f
                text = "dip"
            else:
                dy = np.diff(data.loc[:indexT3, 'Force'], 2)
                fast_indexT2 = np.argmax(dy)
                indexT2 = fast_indexT2 + 1
                text = "no dip"
            break
    forceT2 = np.round(data.loc[indexT2, 'Force'], 2)
    timeT2 = np.round(data.loc[indexT2, 'Time'] - timeT0, 2)

    preload_dosage = np.round(np.trapz(data.loc[indexT0:indexT2, 'Time'], data.loc[indexT0:indexT2, 'Force']), 2)
    thrust_dosage = np.round(np.trapz(data.loc[indexT2:indexT3, 'Time'], data.loc[indexT2:indexT3, 'Force']), 2)
    total_dosage = np.round(preload_dosage + thrust_dosage, 2)

    # Force, time, and index of T1
    if text == "no dip":
        indexT1 = indexT2
    else:
        max_forceT1_index = data.loc[indexT0:indexT2, 'Force'].idxmax()
        # indexT1 = max_forceT1_index + indexT0

        # added myself
        indexT1 = max_forceT1_index

    if indexT2 - indexT1 < 3:
        text = "no dip"

    forceT1 = np.round(data.loc[indexT1, 'Force'], 2)
    timeT1 = np.round(data.loc[indexT1, 'Time'] - timeT0, 2)

    if abs(forceT1 - forceT2) <= 6:
        text = "no dip"

    # Force, time, and index of T4

    s = data['Force']
    indexT4 = s[s != 0].index[-1]
    forceT4 = data.loc[indexT4, 'Force']
    timeT4 = data.loc[indexT4, 'Time'] - timeT0

    timeT2 = np.round(timeT2, 2)

    return timeT0, forceT0, timeT1, forceT1, timeT2, forceT2, indexT2, indexT3, timeT3, forceT3, timeT4, forceT4, preload_dosage, thrust_dosage, total_dosage, text



if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent

    model = joblib.load('svm_classifier_k20.pkl')
    joblib.dump(model, 'svm_classifier_k20.pkl')
    classifier_exp = load_classifier('svm_classifier_k20.pkl')

    profile = Profile()

    app = App(profile, classifier_exp)

    app.mainloop()
