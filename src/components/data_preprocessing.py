import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer





from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

def to_dense(X):
    return X.toarray()


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            Dataset = pd.read_csv("artifacts/Imported_data.csv")
            categorical_features =[
                'project_location',
                #'sustainability',
                'fire_approval_std',
                'head_count_from_client',
                'base_floor_screeding_type',
                'workhall_ceiling_type',
                'door_type_in_rooms',
                'toilets_by_landlord',
                'Fire exits door in SM scope',
                'acs_type',
                'electric_strick_lock_required',
                'access_card_in_scope',
                'access_main_server_in_scope',
                'access_recording_pc_in_scope',
                'critical_room_reader_type',
                'cctv_required_in_open_ofc_corridor',
                'cctv_recording_type',
                'cctv_camera_poe_switch_in_scope',
                'cctv_cloud_license',
                'cctv_monitor_in_scope',
                'bms_in_scope',
                #'bms_additoinal_info',
                #'vms_in_scope',
                'repeater_panel_type',
                #'fire_graphics_required',
                'wld_sytem_type',
                'scope_of_rodent_ctrl_system',
                'vesda_in_scope',
                'vesda_scope',
                'flap_barrier_or_turnstile',
                'scope_of_sprinkler',
                'false_ceiling_void_bracket',
                'scope_of_gas_supression_system',
                #'gss_room_coverage',
                'fe36_supression_system_required',
                'fe36_supression_system_room_coverage',
                'fire_extinguisher_room_coverage',
                #'fire_curtain_type',
                #'lifting_pump_applicable',
                'hvac_config',
                'hvac_24x7_type',
                'dedicated_hvac_room_coverage',
                'vav_in_scope',
                'underdeck_in_scope',
                'toilet_in_scope',
                'workstation_type',
                'workspace_operation_ration',
                'hat_in_scope',
                'usb_in_scope',
                'add_on_rp_soc_in_scope',
                #'ah_netbox_inScope',
                'ups_required',
                'conduit_type',
                'lighting_control_system',
                'it_node_details',
                'node_percent_requirement',
                'backbone_cabling_fiber',
                'pdu_type']
            categorical_categories ={
                'project_location':["NORTH","SOUTH","WEST","SOUTH-CHENNAI","SOUTH-OTHERS"],
                #'sustainability':["None","Leed","Well","Leed&Well"],
                'fire_approval_std':["UL","UL&FM"],
                'head_count_from_client':["YES","NO"],
                'base_floor_screeding_type':["Screeding already done at site. Dismantling of existing cement screed. New cement Screeding for Hardfloor area. Raceways filling and repair",
    "Cement Screed  upto 75mm height with M15 grade concrete",
    "False floor across office Bare cement finish tiles (upto 150 mm height Below carpet area & Cement screeding for Hard floor area)"],
                'workhall_ceiling_type':["Open Ceiling","Closed Ceiling","Partially open and closed ceiling","Open ceiling with Acoustic Paint"],
                'door_type_in_rooms':["Glazed stile doors","Laminated Flush doors with aluminium door frame"],
                'toilets_by_landlord':["Yes","No"],
                'Fire exits door in SM scope':["YES","NO"],
                'acs_type':[ "Lenel","Pro Watch","Software house","HID","Kantech/Pro 3000","Spectra/Essl","Matrix","Spintly"],
                'electric_strick_lock_required':["Yes","No"],
                'access_card_in_scope':["Yes","No"],
                'access_main_server_in_scope':["Yes","No"],
                'access_recording_pc_in_scope':["Yes","No"],
                'critical_room_reader_type':["Card","Biometric"],
                'cctv_required_in_open_ofc_corridor':["NA","Corridor","360°"],
                'cctv_recording_type':["NVR Recording","Server Recording","Cloud Recording(Client scope)","Mile Recording","Genetec Recording"],
                'cctv_camera_poe_switch_in_scope':["PoE Switch Client","PoE Switch SM"],
                'cctv_cloud_license':["Client scope (Cloud license)","SM scope (Cloud license)"],
                'cctv_monitor_in_scope':["Client","SM"],
                'bms_in_scope':["Yes","NA"],
                #'bms_additoinal_info':["NA","IOT Integration","EMS Software","GSM Module","IOT+EMS+GSM","IOT+EMS"],
                #'vms_in_scope':["Yes","NA"],
                'repeater_panel_type':["Active","Passive"],
                #'fire_graphics_required':["NA","Fire Graphics"],
                'wld_sytem_type':[ "NA","Analogue","Addressable"],
                'scope_of_rodent_ctrl_system':["NA","All Critical","All critical+cafe+shaft"],
                'vesda_in_scope':["NA","Yes"],
                'vesda_scope':["NA","Server","Server/Hub","Server/Hub/Lab"],
                'flap_barrier_or_turnstile':["NA","Flap Barrier","Turnstile"],
                'scope_of_sprinkler':["New Sprinkler","Modification"],
                'false_ceiling_void_bracket':["<800",">800"],
                'scope_of_gas_supression_system':["Yes","No"],
                #'gss_room_coverage':["NA","Server","Server&HUB","All Critical room"],
                'fe36_supression_system_required':["Yes","No"],
                'fe36_supression_system_room_coverage':["NA","Server/HUB","UPS+Battery","Hub+UPS+Battery","Server+Hub+UPS+Battery"],
                'fire_extinguisher_room_coverage':["Office","Office+UPS+Battery","Office+Hub+UPS+Battery","Office+Server+Hub+UPS+Battery"],
                #'fire_curtain_type':["NA","BSEN Approved","BLE Approved"],
                #'lifting_pump_applicable':["Yes","NA"],
                'hvac_config':["Lowside","Tap-off+AHU+Lowside","Tap-off+CSU+Lowside","VRF+AHU+Lowside","VRF+CSU+Lowside","VRF+Cassette+Lowside","VRF+Hybrid+Lowside","DX+CSU+Lowside","DX+Cassette+Lowside"],
                'hvac_24x7_type':["None","VRF","DX"],
                'dedicated_hvac_room_coverage':["None","Cafe","Gym","Cafe+Gym","Cafe+Gym+Others"],
                'vav_in_scope':["Yes","No"],
                'underdeck_in_scope':["Yes","No"],
                'toilet_in_scope':["toilet in scope","toilet not in scope"],
                'workstation_type':["Laptop","Laptop+Single Monitor","Laptop+Dual Monitor","Dual Monitor+Dual Eq."],
                'workspace_operation_ration':["0 : 50 : 50","50 : 50 : 0","50 : 20 : 30","50 : 30 : 20","0 : 70 : 30","70 : 20 : 10","100 : 0 : 0","0 : 100 : 0","0 : 0 : 100"],
                'hat_in_scope':["HAT-No","HAT-Yes"],
                'usb_in_scope':[ "1 Port USB-No","1 Port USB-Yes","2 Port USB-No","2 Port USB-Yes"],
                'add_on_rp_soc_in_scope':["Yes","No"],
                #'ah_netbox_inScope':["NA","AH Netbox-Workspace","AH Netbox-Meeting Room","AH Netbox-Workspace+Meeting Room"],
                'ups_required':["100% to Office with N","50% to Office with N","No UPS for Office"],
                'conduit_type':["MS","PVC"],
                'lighting_control_system':["PIR","Daylight Harvest+PIR","LMS-Yes","LMS-No"],
                'it_node_details':["None","CAT-6 1 Node","CAT-6 2 Node","CAT-6 3 Node","CAT-6 4 Node","CAT-6A 1 Node","CAT-6A 2 Node","CAT-6A 3 Node","CAT-6A 4 Node"],
                'node_percent_requirement':["025% (Every 4 W/S)","050% (Every 2 W/S)","075% (Every 3 W/S)","100% (Every W/S)","150% (Every W/S 100% + Alternate W/S 50%)"],
                'backbone_cabling_fiber':["None","OM3 6C","OM3 12C","OM3 24C","OM3 48C","OM4 6C","OM4 12C","OM4 24C","OM4 48C","SM 6C","SM 12C","SM 24C","SM 48C"],
                'pdu_type':["Indian 3pin","IEC","OM3 6C"]
            }
            cat_feature = list(categorical_categories.keys())
            ordinal_features = ['dx_details',
                                'cctv_type',
                                'cctv_mega_pix',
                                'cctv_storage',
                                'fa_type',
                                'pa_system_dx_type',
                                'normal_lane_dx_details',
                                'fire_extinguisher_dx_details',
                                'lighting_design_details',
                                'PHE Fixture Catagory']
            ordinal_categories = {
                'dx_details':["Design Strong Option","Standard Design Option","Cost Effective Design Option"],
                'cctv_type':["ECO","MID","HIGH"],
                'cctv_mega_pix':["2MP","4MP","5MP"],
                'cctv_storage':["30 days Storage","60 days Storage","90 days Storage"],
                'fa_type':["ECO FA","MID FA","HIGH FA"],
                'pa_system_dx_type':["NA","ECO/MID","HIGH"],
                'normal_lane_dx_details':["ECO","MID","HIGH"],
                'fire_extinguisher_dx_details':["ECO/MID","HIGH"],
                'lighting_design_details':["ECO","MID","HIGH"],
                'PHE Fixture Catagory':["NA","ECO","MID","HIGH"]
                }
            
            cat_pipeline = Pipeline([
                    ("Cat_imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder(categories=[categorical_categories[col] for col in categorical_features],handle_unknown = 'ignore', sparse_output = False))])


            
            
            
            ord_pipeline = Pipeline(
                steps = [
                    ("ord_imputer",SimpleImputer(strategy="most_frequent")),
                    ("Ordinal_encoder", OrdinalEncoder(categories=[ordinal_categories[feature] for feature in ordinal_features]))
                ]

            )

            logging.info(f"Categorical_Columns encoding completed: {categorical_features}")
            logging.info(f"The cat features are {len(categorical_features)}")
            logging.info(f"Ordinal_Columns encoding completed: {ordinal_features}")



            preprocessor = ColumnTransformer(

                [
                ("Nominal_pipeline",cat_pipeline,categorical_features),
                ("Ordinal_pipeline",ord_pipeline,ordinal_features)
                ],remainder="passthrough"
            )
            return preprocessor

            
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("The train and test dataset have been imported")
            train_df = train_df.drop("project_name",axis=1)
            test_df = test_df.drop("project_name",axis=1)
            logging.info("removed first column")
            preprocessor_obj = self.get_data_transformer_object()
            logging.info("The preprocessor object has been initiated")
            input_feature_train_df = train_df.iloc[:,:250]
            logging.info(f"shape of input_train {input_feature_train_df.shape}")
            input_feature_test_df = test_df.iloc[:,:250]
            logging.info(f"{input_feature_train_df["sustainability"].unique()}")
            Output_train_df = train_df.iloc[:,250:]
            Output_test_df = test_df.iloc[:,250:]
            SKU_labels = Output_test_df.columns.tolist()
            SKU_label_list = pd.DataFrame(SKU_labels,columns=["SKU_Labels"])
            SKU_label_list.to_csv("/Users/naveen/Desktop/Projects/Space_Matrix/Estimation_Engine/artifacts/SKU_label.csv")
            Output_train_classification_df = Output_train_df.map(lambda x: 1 if x > 0 else x)
            Output_test_classification_df = Output_test_df.map(lambda x:1 if x>0 else x)
            Output_test_classification_df.to_csv("/Users/naveen/Desktop/Projects/Space_Matrix/Estimation_Engine/artifacts/classification_data_output_sample.csv")
            logging.info("Applying preprocessing object on training and test set")
            logging.info(f"Dataset size is {input_feature_train_df.shape}")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            #df = pd.DataFrame(input_feature_train_arr)
            #df.to_csv("/Users/naveen/Desktop/Projects/Space_Matrix/Estimation_Engine/artifacts/train_sample.csv")
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr,np.array(Output_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(Output_test_df)]
            classification_train_arr = np.c_[input_feature_train_arr,np.array(Output_train_classification_df)]
            classification_test_arr = np.c_[input_feature_test_arr,np.array(Output_test_classification_df)]
            logging.info("Saving the object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,classification_train_arr,classification_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
