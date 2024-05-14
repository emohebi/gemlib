import gemlib.framework.knowledgediscovery as kd
import numpy as np


class VisualizationEngine(kd.Discovery):

    def writetofile(self):
        print('\n---------------Discovery----------------\n')

        if self.knowledge.best_sparable_features is not None:
            assert isinstance(self.knowledge.best_sparable_features, list)
            print('best separable features are : "{0}". Please check the plots in the visualization folder.'
                  .format(self.knowledge.best_sparable_features))
            if self.knowledge.bicluster_decision_column is not None:
                df_upper = self.df[self.df[self.knowledge.bicluster_decision_column]]
                df_upper_agg = df_upper.groupby(self.classdefinition.target).count()
                df_upper_agg = df_upper_agg[self.knowledge.bicluster_decision_column]
                df_lower = self.df[~self.df[self.knowledge.bicluster_decision_column]]
                df_lower_agg = df_lower.groupby(self.classdefinition.target).count()
                df_lower_agg = df_lower_agg[self.knowledge.bicluster_decision_column]
                print('based on the decisiosn boundaries we can see that the number of data points above the decision'
                      ' function are as: \n{0} \nand below decision function: \n{1} '
                      .format(df_upper_agg.to_string(), df_lower_agg.to_string()))
                if np.sum(df_lower_agg) < np.sum(df_upper_agg):
                    class_0 = self.classdefinition.classes[0]
                    class_1 = self.classdefinition.classes[1]
                    tot_class_0_rem = df_lower_agg[class_0] * 100.0 / (df_upper_agg.loc[class_0] + df_lower_agg.loc[class_0])
                    tot_class_1_rem = df_lower_agg[class_1] * 100.0 / (df_upper_agg.loc[class_1] + df_lower_agg.loc[class_1])
                    total_removal = np.sum(df_lower_agg) * 100.0 / (np.sum(df_lower_agg) + np.sum(df_upper_agg))
                    print(
                        'therefore, if we remove data points which are below the line then it means we have removed %{0} of'
                        ' {1} and %{2} of {3}. Then total removals would be %{4}[{5}]. Does it make sense?'.format(
                            tot_class_0_rem, class_0, tot_class_1_rem, class_1, total_removal, np.sum(df_lower_agg)
                        ))
                    df_t = self.df[(self.df['TRUTH_ATTENTION_REGION'] == 'dme_attention_on_road')]
                    tp_rate = 100.0 * len(self.df[self.df[self.classdefinition.target] == 'ONRD_TP'])/(len(df_t))
                    #df_t = df_upper[(df_upper['TRUTH_ATTENTION_REGION'] == 'dme_attention_on_road')]
                    tp_rate_new = 100.0 * len(df_upper[df_upper[self.classdefinition.target] == 'ONRD_TP']) / (len(df_t))

                    df_a = self.df[(self.df['TRUTH_ATTENTION_REGION'] != 'dme_attention_on_road')]
                    fp_rate = 100.0 * len(self.df[self.df[self.classdefinition.target] == 'ONRD_FP'])/(len(df_a))
                    #df_a = df_upper[(df_upper['TRUTH_ATTENTION_REGION'] != 'dme_attention_on_road')]
                    fp_rate_new = 100.0 * len(df_upper[df_upper[self.classdefinition.target] == 'ONRD_FP'])/(len(df_a))

                    print(
                        'you may like to know that fp_rate and tp_rate was %{0} and %{1}. After removing those frames '
                        'now fp_rate is %{2} and tp_rate is %{3}'.format(fp_rate, tp_rate, fp_rate_new, tp_rate_new))
