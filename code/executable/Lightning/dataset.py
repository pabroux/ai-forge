import math
import numpy as np
import pandas as pd
import os
import re
import torch
from fnmatch import fnmatch

# DataSet definition


class ESW1DataSet(torch.utils.data.Dataset):
    """
    DataSet for the ESW1 (Emotional Speech in the Wild 1) corpus
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    Args:
        path_input (str): path to audio/feature/input files
        path_gold_standard (str): path to gold-standard files
        max_seq_len (int): max sequence length in seconds. 'None' to deactivate it
        padding (bool): 0 padding to apply. It requires max_len_seq to be set up
        padding_circular (bool): circular padding. It requires padding to be set up
        path_input_pattern (str): audio/feature/input file pattern to import
        path_gold_standard_pattern (str): gold-standard file pattern to import
        input_reader (function): function to read audio/feature/input files. It must return a pandas.Dataframe
        normalize_input (bool): apply a normalization to input (i.e. (input-self.input_mean)/self.input_var)
        normalize_input_assess (bool): assess the mean and the variance of all inputs and store them in self.input_mean and self.input_var
        normalize_target (bool): apply a normalization to target (i.e. target/10)
    """

    def __init__(
        self,
        path_input,
        path_gold_standard,
        max_seq_len=7 * 60,
        padding=True,
        padding_circular=True,
        path_input_pattern="*.csv",
        path_gold_standard_pattern="*.csv",
        input_reader=lambda x: pd.read_csv(x),
        normalize_input=False,
        normalize_input_assess=False,
        normalize_target=False,
    ):
        # Asserting some inputs
        assert (
            max_seq_len is None or max_seq_len > 0
        ), "'max_seq_len' must be None or a positive number."
        assert (
            padding and (max_seq_len is not None) or not padding
        ), "'padding' requires 'max_seq_len' to be set up."
        assert (
            (padding_circular and padding) or (not padding_circular) or not padding
        ), "'padding_circular' requires 'padding' to be set to True."

        # Getting all gold-standard files
        self.gold_standard = []
        gold_standard_dict = {}
        for path, _, files in os.walk(path_gold_standard):
            for name in files:
                if not name.startswith(".") and fnmatch(
                    name, path_gold_standard_pattern
                ):
                    self.gold_standard.append(os.path.join(path, name))
                    extracted_name = re.match(
                        path_gold_standard_pattern.replace("*", "(.*)"), name
                    ).group(1)
                    gold_standard_dict[len(gold_standard_dict)] = extracted_name

        # Getting all input files
        self.input = []
        input_dict = {}
        for path, _, files in os.walk(path_input):
            for name in files:
                if not name.startswith(".") and fnmatch(name, path_input_pattern):
                    self.input.append(os.path.join(path, name))
                    extracted_name = re.match(
                        path_input_pattern.replace("*", "(.*)"), name
                    ).group(1)
                    input_dict[len(input_dict)] = extracted_name

        # Checking whether there is the same number for the input files and the gold-standard ones
        if len(self.gold_standard) != len(self.input):
            raise Exception(
                "Number of input files doesn't match the one of gold-standard files."
            )

        # Checking the matching and sorting the gold-standard files with the input ones
        reversed_gold_standard_dict = {i: j for j, i in gold_standard_dict.items()}
        gold_standard = []
        for i in range(len(input_dict)):
            try:
                gold_standard.append(
                    self.gold_standard[reversed_gold_standard_dict[input_dict[i]]]
                )
            except KeyError:
                raise Exception(
                    f"There is no gold-standard matching for the '{input_dict[i]}' input file."
                )
        self.gold_standard = gold_standard

        # Shareable lists for the code left
        list_csv_input = []
        list_csv_gold = []

        # Checking if input and gold_standard have the same length and assessing the normalization if activated
        list_all_input_tensor = []
        for idx, file_gold in enumerate(gold_standard):
            csv_input = input_reader(self.input[idx])
            csv_gold = pd.read_csv(file_gold, header=None)
            list_csv_input.append(csv_input)
            list_csv_gold.append(csv_gold)
            if (diff_len := (len(csv_gold) - len(csv_input))) != 0:
                error_message = f"The input ({self.input[idx]}) and the gold standard ({file_gold}) don't have the same length."
                if diff_len > 0:
                    error_message += f" The gold standard is longer than the input."
                    raise Exception(error_message)
                elif diff_len < -5:  # Tolerance threshold
                    error_message += f" The input is longer than the gold standard. It exceeds the tolerance threshold 5."
                    raise Exception(error_message)
            if normalize_input_assess:
                list_all_input_tensor.append(torch.tensor(csv_input.values))
        if normalize_input_assess:
            var, mean = torch.var_mean(
                torch.cat(list_all_input_tensor), dim=0, correction=1
            )
            self.input_mean = mean
            self.input_var = var
        else:
            self.input_mean = None
            self.input_var = None

        # Creating the file list according to max_seq_len and checking the timestep among gold-standard files
        ## Each element in the file list is a tuple containing the input/gold_standard index with the start in the ground truth
        if max_seq_len is not None:
            self.file_list_with_max_seq = []
        self.timestep = None
        for idx, csv in enumerate(list_csv_gold):
            if idx == 0:
                self.timestep = csv.iloc[-1, 0] - csv.iloc[-2, 0]
            else:
                if self.timestep != (csv.iloc[-1, 0] - csv.iloc[-2, 0]):
                    raise Exception(f"Not the same timestep among gold-standard files.")
            if max_seq_len is not None:
                nb = math.ceil(
                    math.ceil(csv.iloc[-1, 0] / self.timestep)
                    / int(max_seq_len / self.timestep)
                )
                for y in range(nb):
                    self.file_list_with_max_seq.append(
                        (idx, y * int(max_seq_len / self.timestep))
                    )

        # Other configurations
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.padding_circular = padding_circular
        self.input_reader = input_reader
        self.normalize_input = normalize_input
        self.normalize_target = normalize_target

    def __len__(self):
        """
        Returns the number of samples in our dataset
        """
        return (
            len(self.file_list_with_max_seq)
            if self.max_seq_len is not None
            else len(self.input)
        )

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index idx
        """
        # Reading the gold-standard csv file
        input_and_ground_idx = (
            self.file_list_with_max_seq[idx][0] if self.max_seq_len is not None else idx
        )
        gold_standard_file = self.gold_standard[input_and_ground_idx]
        gold_standard_all = pd.read_csv(gold_standard_file, header=None)
        gold_standard_all = gold_standard_all.iloc[:, 1]

        # Reading the input file
        input_file = self.input[input_and_ground_idx]
        input_all = self.input_reader(input_file)

        # Preparing the final ouput
        diff_len = len(gold_standard_all) - len(input_all)
        # Forcing the same size as the gold_standard
        if diff_len < 0:
            input_all = input_all.iloc[:diff_len]
        gold_standard = gold_standard_all
        input = input_all

        # Cutting if max_seq_len
        if self.max_seq_len:
            start = self.file_list_with_max_seq[idx][1]
            stop = start + int(self.max_seq_len / self.timestep)
            # print(start,stop)
            input = input.iloc[start:stop]
            gold_standard = gold_standard.iloc[start:stop]
            input.reset_index(drop=True, inplace=True)
            gold_standard.reset_index(drop=True, inplace=True)
            # Padding
            if self.padding:
                nb_row_to_add = int(self.max_seq_len / self.timestep) - len(input)
                if nb_row_to_add:
                    input_list = [input]
                    gold_standard_list = [gold_standard]
                    if self.padding_circular:
                        # Circular padding
                        while nb_row_to_add:
                            new_input = input_all.iloc[:nb_row_to_add]
                            new_gold_standard = gold_standard_all.iloc[:nb_row_to_add]
                            input_list.append(new_input)
                            gold_standard_list.append(new_gold_standard)
                            nb_row_to_add -= len(new_input)
                    else:
                        # Zero padding
                        input_list.append(
                            pd.DataFrame(
                                np.zeros((nb_row_to_add, len(input.columns))),
                                columns=input.columns,
                            )
                        )
                        gold_standard_list.append(
                            pd.DataFrame(
                                np.zeros((nb_row_to_add, 1)),
                            )
                        )
                    input = pd.concat(input_list, ignore_index=True)
                    gold_standard = pd.concat(gold_standard_list, ignore_index=True)

        # Tensoring the dataframes
        input = torch.tensor(input.values).float()
        gold_standard = torch.tensor(gold_standard.values).float()

        # Reshaping gold standard
        gold_standard = gold_standard.reshape(-1, 1)

        # Normalizing
        if self.normalize_target:
            gold_standard /= 10
        if self.normalize_input:
            input -= self.input_mean
            input /= self.input_var

        return input, gold_standard
