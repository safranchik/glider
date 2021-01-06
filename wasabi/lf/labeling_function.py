class LabelingFunction:

    def get_vote(self, sample, metadata):
        """
        Casts the votes from the apply() to an integer
        :param sample:
        :param metadata:
        :return:
        """
        vote = self.apply(sample, metadata)

        if vote is None:
            return -1

        return int(vote)

    def apply(self, sample, metadata):
        """
        Contains the logic of the labeling function
        :param sample: data sample to apply labeling function to
        :param metadata: metadata for the dataset
        :return: votes casted for the input sample
        """
        raise NotImplementedError()

    def get_name(self):
        """
        :return: Specified labeling function name
        """
        return type(self).__name__





