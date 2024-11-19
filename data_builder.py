import random
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Define thresholds
energy_threshold = -7.0  # kcal/mol
kd_threshold = 1e-06  # M

prompt_format = """<s>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}: 
{} = seq<{}>

### Response:
{}  = seq<{}>"""



prompt_format_tcr = """<s>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}: 
{} = seq<{}>
The output must be bound to the input: {}

### Response:
{}  = seq<{}>"""

# prompt_format = """<s>[INST] <<SYS>> 
# {}
# <</SYS>> 

# {}:
# {} = seq<{}> [/INST]

# {}  = seq<{}>"""


class Build():
    def __init__(self, tokenizer, path="sequence_sequence.csv", instruction="Drug Peptide Sequence Prediction Tool Based on Protein Target Sequences", prompt_format=prompt_format):
        self.path=path
        self.df = pd.read_csv(path)
        self.instruction = instruction
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format
        self.EOS_TOKEN = self.tokenizer.eos_token
        drug_train, drug_test, target_train, target_test = train_test_split(self.df['chain_a'].index, self.df['chain_b'].index, test_size=0.10)

        self.drug_train = self.df.loc[drug_train, ['chain_a']].to_numpy()
        self.drug_test = self.df.loc[drug_test, ['chain_a']].to_numpy()
        self.target_train = self.df.loc[target_train, ['chain_b']].to_numpy()
        self.target_test = self.df.loc[target_test, ['chain_b']].to_numpy()
        
        # self.labels_train = (self.df.loc[drug_train, ['energy']] <= energy_threshold).to_numpy() & (self.df.loc[drug_train, ['KdM']] <= kd_threshold).to_numpy()
        # self.labels_test = (self.df.loc[drug_test, ['energy']] <= energy_threshold).to_numpy() & (self.df.loc[drug_test, ['KdM']] <= kd_threshold).to_numpy()
        
        # self.labels_train = self.labels_train.astype(int)
        # self.labels_test = self.labels_test.astype(int)
        
        zipped_target_drug_train = list(zip(self.target_train, self.drug_train))
        zipped_target_drug_test = list(zip(self.target_test, self.drug_test))

        self.data_train = self.preprocessor(zipped_target_drug_train)
        self.data_test = self.preprocessor(zipped_target_drug_test)
        

        self.hf_dataset_train = Dataset.from_dict(self.data_train)
        self.hf_dataset_test = Dataset.from_dict(self.data_test)
        
        
    def preprocessor(self, examples):
        texts = [
            self.prompt_format.format(self.instruction, "Generate a potent drug peptide sequence for protein target sequence", item[0], item[1]) + self.EOS_TOKEN for item in examples
        ]
        
        return {
            "text":texts
            }


class BuildUniversally():
    def __init__(self, tokenizer, paths=["sequence_sequence.csv",], prompt_format=prompt_format):
    # def __init__(self, tokenizer, paths=["sequence_sequence.csv", "kd.csv"], prompt_format=prompt_format):
    # def __init__(self, tokenizer=None, paths=["UniRef50"], prompt_format=prompt_format):
        self.paths=paths
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format
        self.EOS_TOKEN = self.tokenizer.eos_token
        
        
        for path in paths:
            if path == "UniRef50":
                self.dataset = load_dataset("agemagician/uniref50")
                print(self.dataset)
                
            if path == "sequence_sequence.csv":
        
                self.df = pd.read_csv(path)
                self.instruction ="Drug Peptide Sequence Prediction Tool Based on Protein Target Sequences"
                drug_train, drug_test, target_train, target_test = train_test_split(self.df['chain_a'].index, self.df['chain_b'].index, test_size=0.20)

                self.drug_train = self.df.loc[drug_train, ['chain_a']].to_numpy()
                self.drug_test = self.df.loc[drug_test, ['chain_a']].to_numpy()
                self.target_train = self.df.loc[target_train, ['chain_b']].to_numpy()
                self.target_test = self.df.loc[target_test, ['chain_b']].to_numpy()
                
                
                # self.labels_train = (self.df.loc[drug_train, ['energy']] <= energy_threshold).to_numpy() & (self.df.loc[drug_train, ['KdM']] <= kd_threshold).to_numpy()
                # self.labels_test = (self.df.loc[drug_test, ['energy']] <= energy_threshold).to_numpy() & (self.df.loc[drug_test, ['KdM']] <= kd_threshold).to_numpy()
                
                # self.labels_train = self.labels_train.astype(int)
                # self.labels_test = self.labels_test.astype(int)
                
                zipped_target_drug_train = list(zip(self.target_train, self.drug_train))
                zipped_target_drug_test = list(zip(self.target_test, self.drug_test))

                self.data_train_seq_seq = self.preprocessor(zipped_target_drug_train, self.instruction, "Generate a potent drug peptide sequence for protein sequence", "protein sequence", "drug peptide sequence")
                self.data_test_seq_seq = self.preprocessor(zipped_target_drug_test, self.instruction, "Generate a potent drug peptide sequence for protein sequence", "protein sequence", "drug peptide sequence")
                

                # self.hf_dataset_train = Dataset.from_dict(self.data_train)
                # self.hf_dataset_test = Dataset.from_dict(self.data_test)
            if path == "kd.csv":
                self.df = pd.read_csv(path)
                # self.df = self.df.sort_values(by='Kd (nM)')
                # self.df = self.df.iloc[:5000]
                
                
                drug_train, drug_test, target_train, target_test = train_test_split(self.df['Ligand SMILES'].index, self.df['BindingDB Target Chain Sequence'].index, test_size=0.20)

                self.drug_train = self.df.loc[drug_train, ['Ligand SMILES']].to_numpy()
                self.drug_test = self.df.loc[drug_test, ['Ligand SMILES']].to_numpy()
                self.target_train = self.df.loc[target_train, ['BindingDB Target Chain Sequence']].to_numpy()
                self.target_test = self.df.loc[target_test, ['BindingDB Target Chain Sequence']].to_numpy()
                
                # self.labels_train = (self.df.loc[drug_train, ['energy']] <= energy_threshold).to_numpy() & (self.df.loc[drug_train, ['KdM']] <= kd_threshold).to_numpy()
                # self.labels_test = (self.df.loc[drug_test, ['energy']] <= energy_threshold).to_numpy() & (self.df.loc[drug_test, ['KdM']] <= kd_threshold).to_numpy()
                
                # self.labels_train = self.labels_train.astype(int)
                # self.labels_test = self.labels_test.astype(int)
                
                zipped_target_drug_train = list(zip(self.target_train[:len(self.target_train)//2], self.drug_train[:len(self.target_train)//2]))
                zipped_target_drug_test = list(zip(self.target_test[:len(self.target_test)//2], self.drug_test[:len(self.target_test)//2]))

                self.data_train_seq_smiles = self.preprocessor(zipped_target_drug_train, "Protein Target-Based SMILES Ligand Predictor Tool", "Generate a potent ligand for protein target sequence")
                self.data_test_seq_smiles = self.preprocessor(zipped_target_drug_test, "Protein Target-Based SMILES Ligand Predictor Tool", "Generate a potent ligand for protein target sequence")

                zipped_drug_target_train = list(zip(self.drug_train[len(self.target_train)//2:], self.target_train[len(self.target_train)//2:]))
                zipped_drug_target_test = list(zip(self.drug_test[len(self.target_test)//2:], self.target_test[len(self.target_test)//2:]))

                self.data_train_smiles_seq = self.preprocessor(zipped_drug_target_train, "SMILES Ligand-Based Protein Target Predictor Tool", "Generate a potent protein target sequence for SMILES ligand")
                self.data_test_smiles_seq = self.preprocessor(zipped_drug_target_test, "SMILES Ligand-Based Protein Target Predictor Tool", "Generate a potent protein target sequence for SMILES ligand")


        self.universal_dataset_train = self.data_train_seq_seq
        # self.universal_dataset_train['text'].extend(self.data_train_seq_smiles['text'])
        # self.universal_dataset_train['text'].extend(self.data_train_smiles_seq['text'])
        random.shuffle(self.universal_dataset_train['text'])

        self.universal_dataset_test = self.data_test_seq_seq
        # self.universal_dataset_test['text'].extend(self.data_test_seq_smiles['text'])
        # self.universal_dataset_test['text'].extend(self.data_test_smiles_seq['text'])
        random.shuffle(self.universal_dataset_test['text'])

        test_len = len(self.universal_dataset_test['text'])
        self.universal_dataset_validation = {}
        self.universal_dataset_validation['text'] = self.universal_dataset_test['text'][:test_len]
        # self.universal_dataset_test['text'] = self.universal_dataset_test['text'][test_len//2:]
        
        self.hf_dataset_train = Dataset.from_dict(self.universal_dataset_train)
        self.hf_dataset_validation = Dataset.from_dict(self.universal_dataset_validation)
        self.hf_dataset_test = Dataset.from_dict(self.universal_dataset_test)
        
        
        
    def preprocessor(self, examples, instruction, initial_prompt, input_, output_):
        texts = [
            self.prompt_format.format(instruction, initial_prompt, input_, item[0][0], output_, item[1][0]) + self.EOS_TOKEN for item in examples
        ]
        
        return {
            "text":texts
            }



class BuildTCREpitope():
    def __init__(self, tokenizer, prompt_format=prompt_format_tcr):
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format
        self.EOS_TOKEN = self.tokenizer.eos_token
        self.path_train = './tcr_sequence_sequence_train.csv'
        self.path_test = './tcr_sequence_sequence_test.csv'
        self.path_valid = './tcr_sequence_sequence_valid.csv'
        

        self.df_train = pd.read_csv(self.path_train)
        self.df_test = pd.read_csv(self.path_test)
        self.df_valid = pd.read_csv(self.path_valid)
        self.instruction ="TCR Peptide Sequence Prediction Tool Based on epitope Target Sequences"

        self.drug_train = self.df_train[['tcr_full']].to_numpy()
        self.drug_test = self.df_test[['tcr_full']].to_numpy()
        self.drug_valid = self.df_valid[['tcr_full']].to_numpy()
        
        self.target_train = self.df_train[['epitope_aa']].to_numpy()
        self.target_test = self.df_test[['epitope_aa']].to_numpy()
        self.target_valid = self.df_valid[['epitope_aa']].to_numpy()
        
        self.label_train = self.df_train[['label']].to_numpy()
        self.label_test = self.df_test[['label']].to_numpy()
        self.label_valid = self.df_valid[['label']].to_numpy()
        
        
        zipped_target_drug_train = list(zip(self.target_train, self.drug_train, self.label_train))
        zipped_target_drug_test = list(zip(self.target_test, self.drug_test, self.label_test))
        zipped_target_drug_valid = list(zip(self.target_valid, self.drug_valid, self.label_valid))

        self.data_train_seq_seq = self.preprocessor(zipped_target_drug_train, self.instruction, "Generate a potent TCR peptide sequence for epitope sequence", "epitope sequence", "TCR peptide sequence")
        self.data_test_seq_seq = self.preprocessor(zipped_target_drug_test, self.instruction, "Generate a potent TCR peptide sequence for epitope sequence", "epitope sequence", "TCR peptide sequence")
        self.data_valid_seq_seq = self.preprocessor(zipped_target_drug_valid, self.instruction, "Generate a potent TCR peptide sequence for epitope sequence", "epitope sequence", "TCR peptide sequence")


        self.universal_dataset_train = self.data_train_seq_seq
        random.shuffle(self.universal_dataset_train['text'])

        self.universal_dataset_test = self.data_test_seq_seq
        random.shuffle(self.universal_dataset_test['text'])
        
        self.universal_dataset_valid = self.data_valid_seq_seq
        random.shuffle(self.universal_dataset_valid['text'])
        
        self.hf_dataset_train = Dataset.from_dict(self.universal_dataset_train)
        self.hf_dataset_validation = Dataset.from_dict(self.universal_dataset_valid)
        self.hf_dataset_test = Dataset.from_dict(self.universal_dataset_test)
        
        
        
    def preprocessor(self, examples, instruction, initial_prompt, input_, output_):
        texts = [
            self.prompt_format.format(instruction, initial_prompt, input_, item[0][0], bool(item[2][0]), output_, item[1][0]) + self.EOS_TOKEN for item in examples
        ]
        
        return {
            "text":texts
            }
        
        
if __name__ == "__main__":
    BuildUniversally()