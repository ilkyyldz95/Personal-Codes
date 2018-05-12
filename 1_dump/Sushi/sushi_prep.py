import xlrd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class sushi_prep(object):
    def __init__(self, location='./'):
        self.location = location
        self.no_of_items = 100
        self.no_of_features = 18
        self.no_of_users = 5000

    def get_features(self):
        '''
        2. style, 0 or 1
        3. major group, 0 or 1
        4. minor group (0-11, categorical)
        5. the heaviness/oiliness in taste, range[0-4] 0:heavy/oily
        6. how frequently the user eats the SUSHI, range[0-3] 3:frequently eat
        7. normalized price
        8. how frequently the SUSHI is sold in sushi shop, range[0-1] 1:the most frequently
        '''
        # file directory
        file_loc = self.location + 'features.xlsx'
        wkb = xlrd.open_workbook(file_loc)
        sheet = wkb.sheet_by_index(0)
        # construct feature matrix except for minor group
        feature_matrix = np.zeros([self.no_of_items, 6])
        for row in range(sheet.nrows):
            i = 0
            for col in list(range(2, 4)) + list(range(5, 9)):
                feature_matrix[row, i] = sheet.cell_value(row, col)
                i += 1
        # make the minor group feature categorical
        minor_group_matrix = np.zeros([self.no_of_items, 12])
        for row in range(sheet.nrows):
            minor_group = int(sheet.cell_value(row, 4))
            minor_group_matrix[row, minor_group] = 1
        # construct the whole feature matrix
        return np.concatenate([feature_matrix, minor_group_matrix], axis=-1)

    def create_base_network(self, no_of_layers=2, max_no_of_nodes=18):
        '''
        Base network of siamese for feature extraction
        :return:
        '''
        base_net = Sequential()
        base_net.add(Dense(max_no_of_nodes, input_shape=(self.no_of_features,), activation='tanh'))
        for i in range(no_of_layers-1):
            base_net.add(Dense(int(max_no_of_nodes / (i+1.5)), activation='tanh'))
        return base_net

    def get_similar_users(self, user_sim_thr=0):
        # file directory
        file_loc = self.location + './sushi3.udata'
        # get rid of the user index
        ranking_matrix = np.genfromtxt(file_loc, delimiter='\t')[:, 1:]
        # normalize features
        for col in range(ranking_matrix.shape[1]):
            ranking_matrix[:, col] = 1.0 * ranking_matrix[:, col] / np.max(ranking_matrix[:, col])
        distance_matrix = np.zeros((self.no_of_users, self.no_of_users))
        for user1 in range(self.no_of_users):
            for user2 in np.arange(user1 + 1, self.no_of_users):
                feature1 = ranking_matrix[user1]
                feature2 = ranking_matrix[user2]
                distance_matrix[user1][user2] = np.linalg.norm(feature1 - feature2)
        # normalize distances
        distance_matrix = 1.0 * distance_matrix / np.max(distance_matrix)
        # separate users into similar(1) or not(0)
        for user1 in range(self.no_of_users):
            for user2 in np.arange(user1 + 1, self.no_of_users):
                distance_matrix[user1][user2] = int(distance_matrix[user1][user2] > user_sim_thr)
                distance_matrix[user2][user1] = distance_matrix[user1][user2]
        # find the largest cluster of similar users
        summed = np.sum(distance_matrix, axis=-1)
        return np.where(distance_matrix[np.argmax(summed), :] == 1)[0]

    def get_pairwise_comparisons(self, no_of_labelers=5000, sample_per_user=10):
        '''
        Each row in the data is the ranking of 10 items out of 100 items
        Format: 0<sp><10><sp><1st item ID>....<10-th item ID><nl>
        :return: label array
        each row has: item1, item2, y_ij
        item1: 0-99
        item2: 0-99
        y_ij: +1 or -1
        '''
        np.random.seed(1)
        # file directory
        file_loc = self.location + 'sushi3b.5000.10.order'
        # get rid of the first two elements of each row
        ranking_matrix = np.genfromtxt(file_loc, delimiter=' ')[:, 2:]
        # generate pairwise comparisons from rankings
        # randomly choose labelers and their pairwise comparisons
        users = np.random.choice(self.no_of_users, no_of_labelers, replace=False)
        pairs = np.transpose([np.tile(range(10), 10), np.repeat(range(10), 10)])
        comparisons = []
        for row in users:
            # pairs of ranking positions
            pair_indices = np.random.choice(100, sample_per_user, replace=False)
            rank_positions_1 = pairs[pair_indices, 0]
            rank_positions_2 = pairs[pair_indices, 1]
            for comp_ind in range(sample_per_user):
                item1 = int(ranking_matrix[row, rank_positions_1[comp_ind]])
                item2 = int(ranking_matrix[row, rank_positions_2[comp_ind]])
                # get comparison label, the top item is the most preferred one
                y_ij = 2 * int(rank_positions_1[comp_ind] <= rank_positions_2[comp_ind]) - 1
                comparisons += [(item1, item2, y_ij)]
        return np.array(comparisons)

    def sushi_comp_training_data(self, kthFold, no_of_labelers=5000, sample_per_user=10):
        '''
        return the feature pairs and comparison labels except for kthFold
        :return:
        features: array, no_of_comparisons by 18
        labels: array, no_of_comparisons by 1
        '''
        np.random.seed(1)
        all_features = self.get_features()
        all_comparisons = self.get_pairwise_comparisons(no_of_labelers=no_of_labelers, sample_per_user=sample_per_user)
        # find fold indices
        fold_size = int(len(all_comparisons) / 5)
        all_indices = np.random.permutation(len(all_comparisons))
        ind_train = np.concatenate([all_indices[0:kthFold * fold_size], all_indices[(kthFold + 1) * fold_size:]])
        # find corresponding features and labels
        features_1 = all_features[all_comparisons[ind_train,0]]
        features_2 = all_features[all_comparisons[ind_train,1]]
        labels = all_comparisons[ind_train,2]
        return features_1, features_2, labels

    def sushi_comp_validation_data(self, kthFold, no_of_labelers=5000, sample_per_user=10):
        '''
        return the feature pairs and comparison labels for kthFold
        :return:
        features: array, no_of_comparisons by 18
        labels: array, no_of_comparisons by 1
        '''
        np.random.seed(1)
        all_features = self.get_features()
        all_comparisons = self.get_pairwise_comparisons()
        # find fold indices
        fold_size = int(len(all_comparisons) / 5)
        all_indices = np.random.permutation(len(all_comparisons))
        ind_val = all_indices[kthFold * fold_size: (kthFold + 1) * fold_size]
        # find corresponding features and labels
        features_1 = all_features[all_comparisons[ind_val, 0]]
        features_2 = all_features[all_comparisons[ind_val, 1]]
        labels = all_comparisons[ind_val, 2]
        return features_1, features_2, labels

    def get_absolute_labels(self, no_of_labelers=5000, sample_per_user=10):
        '''***********************************************************
        - matrix style data separated by <sp>
        - each row corresponds to the user index
        - each column corresponds to the SUSHI in the item set B
        - using five-point-scale, 0:the most disliked, 4:the most preferred, -1:not rated
        - sample_per_user can be max 10
        :return: label array
        each row has: (item_i, y_i)
        item_i: 0-99
        y_i: binary absolute label. score threshold depends on the user to suppress the effect of bias
        '''
        np.random.seed(1)
        # file directory
        file_loc = self.location + 'sushi3b.5000.10.score'
        score_matrix = np.genfromtxt(file_loc, delimiter=' ')
        # get existing absolute labels
        # randomly choose labelers and their absolute labels
        users = np.random.choice(self.no_of_users, no_of_labelers, replace=False)
        absolute_labels = []
        for row in users:
            items = np.where(score_matrix[row] != -1)[0]
            score_thr = np.mean(score_matrix[row, items])
            if sample_per_user < 10:
                items = np.random.choice(items, sample_per_user, replace=False)
            for item in items:
                y_i = int(score_matrix[row, int(item)] >= score_thr)
                absolute_labels += [(int(item), y_i)]
        return np.array(absolute_labels)

    def sushi_abs_training_data(self, kthFold, no_of_labelers=5000, sample_per_user=10):
        '''
        return features and absolute labels except for kthFold
        :return:
        feature: array, no_of_absolute_labels by 18
        label: array, no_of_absolute_labels by 5
        '''
        np.random.seed(1)
        all_features = self.get_features()
        all_abs_labels = self.get_absolute_labels(no_of_labelers=no_of_labelers, sample_per_user=sample_per_user)
        # find fold indices
        fold_size = int(len(all_abs_labels) / 5)
        all_indices = np.random.permutation(len(all_abs_labels))
        ind_train = np.concatenate([all_indices[0:kthFold * fold_size], all_indices[(kthFold + 1) * fold_size:]])
        # find corresponding features and labels
        features = all_features[all_abs_labels[ind_train,0]]
        labels = all_abs_labels[ind_train,1:]
        return features, labels

    def sushi_abs_validation_data(self, kthFold, no_of_labelers=5000, sample_per_user=10):
        '''
        return features and absolute labels for kthFold
        :return:
        feature: array, no_of_absolute_labels by 18
        label: array, no_of_absolute_labels by 5
        '''
        np.random.seed(1)
        all_features = self.get_features()
        all_abs_labels = self.get_absolute_labels()
        # find fold indices
        fold_size = int(len(all_abs_labels) / 5)
        all_indices = np.random.permutation(len(all_abs_labels))
        ind_val = all_indices[kthFold * fold_size: (kthFold + 1) * fold_size]
        # find corresponding features and labels
        features = all_features[all_abs_labels[ind_val,0]]
        labels = all_abs_labels[ind_val,1:]
        return features, labels

    def get_combined_labels(self, no_of_labelers=5000, sample_per_user=10, user_sim_thr=0):
        '''
        first choose which users to get data from, then choose the number of pairwise comparisons per user
        :return:
        each row has: (item_i, item_j, y_ij, y_i)
        item: 0-99
        y_i: binary absolute label. score threshold depends on the user to suppress the effect of bias
        y_ij: -1 or 1
        user_sim_thr: use only similar users if not 0
        '''
        np.random.seed(1)
        # file directory
        score_matrix = np.genfromtxt(self.location + 'sushi3b.5000.10.score', delimiter=' ')
        ranking_matrix = np.genfromtxt(self.location + 'sushi3b.5000.10.order', delimiter=' ')[:, 2:]
        # choose users
        if user_sim_thr > 0:
            users = self.get_similar_users(user_sim_thr=user_sim_thr)
        else:
            users = np.random.choice(self.no_of_users, no_of_labelers, replace=False)

        print(users.shape)
        print(users)

        pairs = np.transpose([np.tile(range(10), 10), np.repeat(range(10), 10)])
        combined_labels = []
        for row in users:
            # find score threshold for this user
            score_thr = np.mean(score_matrix[row, np.where(score_matrix[row] != -1)[0]])
            # pairs of ranking positions
            pair_indices = np.random.choice(100, sample_per_user, replace=False)
            rank_positions_1 = pairs[pair_indices, 0]
            rank_positions_2 = pairs[pair_indices, 1]
            for comp_ind in range(sample_per_user):
                item1 = int(ranking_matrix[row, rank_positions_1[comp_ind]])
                item2 = int(ranking_matrix[row, rank_positions_2[comp_ind]])
                # get absolute labels
                y_i = int(score_matrix[row, int(item1)] >= score_thr)
                # get comparison label, the top item is the most preferred one
                y_ij = 2 * int(rank_positions_1[comp_ind] <= rank_positions_2[comp_ind]) - 1
                combined_labels += [(item1, item2, y_ij, y_i)]
        return np.array(combined_labels)

    def sushi_combined_training_data(self, kthFold, no_of_labelers=5000, sample_per_user=10, user_sim_thr=0):
        np.random.seed(1)
        all_features = self.get_features()
        all_combined_labels = self.get_combined_labels(no_of_labelers=no_of_labelers, sample_per_user=sample_per_user,
                                                       user_sim_thr=user_sim_thr)
        # find fold indices
        fold_size = int(len(all_combined_labels) / 5)
        all_indices = np.random.permutation(len(all_combined_labels))
        ind_train = np.concatenate([all_indices[0:kthFold * fold_size], all_indices[(kthFold + 1) * fold_size:]])
        # find corresponding features and labels
        features_1 = all_features[all_combined_labels[ind_train, 0]]
        features_2 = all_features[all_combined_labels[ind_train, 1]]
        comp_labels = all_combined_labels[ind_train, 2]
        # choose unique absolute labels
        unique_item_indices = np.unique(all_combined_labels[ind_train, 0], return_index=True)[1]
        features = all_features[all_combined_labels[unique_item_indices, 0]]
        abs_labels = all_combined_labels[unique_item_indices, 3:]
        return features_1, features_2, comp_labels, features, abs_labels

    def sushi_combined_validation_data(self, kthFold, no_of_labelers=5000, sample_per_user=10, user_sim_thr=0):
        np.random.seed(1)
        all_features = self.get_features()
        all_combined_labels = self.get_combined_labels(user_sim_thr=user_sim_thr)
        # find fold indices
        fold_size = int(len(all_combined_labels) / 5)
        all_indices = np.random.permutation(len(all_combined_labels))
        ind_val = all_indices[kthFold * fold_size: (kthFold + 1) * fold_size:]
        # find corresponding features and labels
        features_1 = all_features[all_combined_labels[ind_val, 0]]
        features_2 = all_features[all_combined_labels[ind_val, 1]]
        comp_labels = all_combined_labels[ind_val, 2]
        # choose unique absolute labels
        unique_item_indices = np.unique(all_combined_labels[ind_val, 0], return_index=True)[1]
        features = all_features[all_combined_labels[unique_item_indices, 0]]
        abs_labels = all_combined_labels[unique_item_indices, 3:]
        return features_1, features_2, comp_labels, features, abs_labels










