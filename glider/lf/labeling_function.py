class LabelingFunction:

    def apply(self, context):
        """
        Contains the logic of the labeling function.
        :param context: contextual information to help users generate a vote for the given sample.
        :return: votes casted for the input sample.
        """
        raise NotImplementedError()

    def get_name(self):
        """
        :return: Specified labeling function name.
        """
        return type(self).__name__


class SyntheticLabelingFunction:
    """
    Synthetic labeling function that has access to labels.
    """

    def apply(self, context, label):
        """
        Contains the logic of the labeling function that accesses the sample label.
        :param context: contextual information to help users generate a vote for the given sample.
        :param label: label for given sample.
        :return: votes casted for the input sample.
        """
        raise NotImplementedError()

    def get_name(self):
        """
        :return: Specified labeling function name.
        """
        return type(self).__name__

