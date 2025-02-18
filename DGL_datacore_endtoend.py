import os
import csv
import glob
import json
import random
import numpy as np
import sys
import dgl
import networkx as nx
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from scipy.optimize import linear_sum_assignment

# Define the custom dataset class
class GraphDataset(Dataset):
    def __init__(self, dataset="NC2017_Dev_Ver1_Img", ratio):
        self.root = "/hdd/Dataset/"
        self.dataset_name = dataset
        self.dataset = os.path.join(self.root, dataset)
        self.reference = os.path.join(self.dataset, "reference/provenance/")
        self.world = os.path.join(self.dataset, "world/")

        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                            ])
        self.ratio = ratio
        self.data = self.build_data()
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
       
        try:
            item = self.data[idx]
            undirected_path_matrix, directed_path_matrix, image_list = item[0], item[1], item[2]
            images = []
            for image_path in image_list:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
                image = (image - image.min()) / (image.max() - image.min())
                images.append(image)
            images_tensor = torch.stack(images)

        except:
            return self.__getitem__((idx + 1) % len(self)) 
            
        return undirected_path_matrix, directed_path_matrix, images_tensor
    
   
    
    def get_rows_from_csv(self, file_path):
        with open(file_path, 'r') as csv_file:
            # TaskID|ProvenanceProbeFileID|ProvenanceProbeFileName|BaseFileName|BaseBrowserFileName|JournalName|JournalFileName|JournalMD5
            reader = csv.DictReader(csv_file, delimiter='|')
            #column_names = reader.fieldnames
            rows = []
            for row in reader:
                rows.append(row)
        return rows

    def read_ref(self):
        ref_file = glob.glob(self.reference + "*ref.csv")[0]
        rows = self.get_rows_from_csv(ref_file)
        return rows
        #['TaskID', 'ProvenanceProbeFileID', 'ProvenanceProbeFileName', 'BaseFileName', 'BaseBrowserFileName', 'JournalName', 'JournalFileName', 'JournalMD5']

    def read_ref_node(self):
        ref_file = glob.glob(self.reference + "*ref-node.csv")[0]
        rows = self.get_rows_from_csv(ref_file)
        return rows
        #['ProvenanceProbeFileID', 'WorldFileID', 'WorldFileName', 'JournalNodeID']

    def get_valid_journal(self,ProvenanceProbeFileID):
        ref_rows = self.read_ref()
        journal_json = ''
        for row in ref_rows:
            if row['ProvenanceProbeFileID'] == ProvenanceProbeFileID:
                journal_json = row['JournalMD5']
                break
        journal_json_path = os.path.join(self.reference, 'journals', journal_json+'.json')

        journal_nodes_id_name = {}
        ref_node_rows = self.read_ref_node()
        for row in ref_node_rows:
            if row['ProvenanceProbeFileID'] == ProvenanceProbeFileID:
                journal_nodes_id_name[row['JournalNodeID']] = row['WorldFileName']


        with open(journal_json_path, 'r') as file:
            json_data = json.load(file)
            all_nodes = json_data["nodes"]
            all_links = json_data["links"]
            valid_nodes_index_path = {}
            valid_links = []

            for index, node in enumerate(all_nodes):
                if node['id'] in journal_nodes_id_name.keys():
                    image_name = journal_nodes_id_name[node['id']][6:]
                    image_name = self.world + image_name
                    image_path  = os.path.join(self.dataset, image_name)
                    if image_path.endswith(".bmp") or image_path.endswith(".tif") or image_path.endswith(".jpeg") or image_path.endswith(".nef") or image_path.endswith(".raf") or image_path.endswith(".dng"):
                        image_path = image_path.split('.')[0]+".jpg"          
                    elif image_path.endswith(".jpg") or image_path.endswith(".png"):
                        pass
                    else:
                        continue
                    
                    valid_nodes_index_path[index] = image_path

            for link in all_links:
                if link['source'] in valid_nodes_index_path.keys() and link['target'] in valid_nodes_index_path.keys():
                    valid_links.append({"source":link["source"],"target":link["target"]})

        return valid_nodes_index_path, valid_links
    

    def get_graph(self,valid_nodes_index_path, valid_links):
        valid_nodes = np.array(list(valid_nodes_index_path.keys()))
        node_mapping = {orig: cont for cont, orig in enumerate(valid_nodes)}
        reverse_node_mapping = {cont:orig for cont, orig in enumerate(valid_nodes)}
        src_nodes = []
        dst_nodes = []
        for link in valid_links:
            if link["source"] not in valid_nodes or link["target"] not in valid_nodes:
                valid_links.remove(link)
                continue
            src_nodes.append(link["source"])
            dst_nodes.append(link["target"])

        # Apply the mapping
        src_mapped = np.array([node_mapping[node] for node in src_nodes])
        dst_mapped = np.array([node_mapping[node] for node in dst_nodes])
        g = dgl.graph((src_mapped, dst_mapped))
        
        # Convert the DGL graph to a NetworkX graph
        
        D_nx_g = g.to_networkx().to_directed()
        UD_nx_g = g.to_networkx().to_undirected()
        
        # Compute shortest path lengths
        D_path_lengths = dict(nx.all_pairs_shortest_path_length(D_nx_g))
        UD_path_lengths = dict(nx.all_pairs_shortest_path_length(UD_nx_g))
        #print(path_lengths)

        # Number of nodes
        n = g.number_of_nodes()

        # Initialize the matrix with infinity where there is no path
        directed_path_matrix = np.full((n, n), fill_value=-1.0)
        undirected_path_matrix = np.full((n, n), fill_value=100)

        # Fill the matrix with the shortest path lengths
        for i in range(n):
            for j in range(n):
                if j in D_path_lengths[i] and i != j:
                    directed_path_matrix[i][j] = D_path_lengths[i][j]
                    #path_matrix[j][i] = path_lengths[i][j]
                if j in UD_path_lengths[i] and i != j:
                    if UD_path_lengths[i][j] == 1:
                        undirected_path_matrix[i][j] = UD_path_lengths[i][j]
                    else:
                        undirected_path_matrix[i][j] = UD_path_lengths[i][j]
                    #path_matrix[j][i] = path_lengths[i][j]

        # Set diagonal to 0
        np.fill_diagonal(directed_path_matrix, 0)
        np.fill_diagonal(undirected_path_matrix, 0)
        
        directed_path_matrix = torch.from_numpy(directed_path_matrix)
        undirected_path_matrix = torch.from_numpy(undirected_path_matrix)

        image_list = []
        for i in range(len(directed_path_matrix)):
            index = reverse_node_mapping[i]
            image_list.append(valid_nodes_index_path[index])


        return  undirected_path_matrix, directed_path_matrix, image_list

        

    def get_data_func(self,probe_ID):
        data = []
        try:
            valid_nodes_index_path, valid_links = self.get_valid_journal(probe_ID)
            if valid_nodes_index_path and valid_links:
                undirected_path_matrix, directed_path_matrix, image_list = self.get_graph(valid_nodes_index_path, valid_links)
                data.append((undirected_path_matrix, directed_path_matrix, image_list))
            return data
        except Exception as e:
            print(e)

    def random_remove_node(valid_nodes_index_path, valid_links):
        nodes = valid_nodes_index_path.keys()
        random_node = random.choice(list(nodes))
        result_links = valid_links
        for link in result_links:
            if random_node in link:
                result_links.remove(link)
        if result_links:
            return result_links

    def build_data(self):
        data_rows = self.read_ref()

        probe_IDs = []
        journals = []
    
        for id, row in enumerate(data_rows):
            probe_ID = row['ProvenanceProbeFileID']
            if row['JournalMD5'] not in journals:
                probe_IDs.append(probe_ID)
                journals.append(row['JournalMD5'])

        max_len = len(probe_IDs)
        probe_IDs = probe_IDs[0:int(max_len*self.ratio[0])]

        data = []
        for probe_ID in probe_IDs:
            result = self.get_data_func(probe_ID)
            data.extend(result)
        
        return data
    
if __name__ == "__main__":
    dataset_name = "NC2017_Dev_Ver1_Img"
    dataset = GraphDataset(dataset=dataset_name,ratio=(0.7, 0.3))
    print(len(dataset))


