from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    template="generate sort and simple note on following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="generate 5 short quiz questions on following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge the provided short note and quiz \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallelChain = RunnableParallel({
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
})

mergeChain = prompt3 | model1 | parser

chain = parallelChain | mergeChain

text = """
1.4. Support Vector Machines
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.

1.4.1. Classification
SVC, NuSVC and LinearSVC are classes capable of performing binary and multi-class classification on a dataset.

../_images/sphx_glr_plot_iris_svc_001.png
SVC and NuSVC are similar methods, but accept slightly different sets of parameters and have different mathematical formulations (see section Mathematical formulation). On the other hand, LinearSVC is another (faster) implementation of Support Vector Classification for the case of a linear kernel. It also lacks some of the attributes of SVC and NuSVC, like support_. LinearSVC uses squared_hinge loss and due to its implementation in liblinear it also regularizes the intercept, if considered. This effect can however be reduced by carefully fine tuning its intercept_scaling parameter, which allows the intercept term to have a different regularization behavior compared to the other features. The classification results and score can therefore differ from the other two classifiers.

As other classifiers, SVC, NuSVC and LinearSVC take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, and an array y of class labels (strings or integers), of shape (n_samples):

from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
SVC()
After being fitted, the model can then be used to predict new values:

clf.predict([[2., 2.]])
array([1])
SVMs decision function (detailed in the Mathematical formulation) depends on some subset of the training data, called the support vectors. Some properties of these support vectors can be found in attributes support_vectors_, support_ and n_support_:

# get support vectors
clf.support_vectors_
array([[0., 0.],
       [1., 1.]])
# get indices of support vectors
clf.support_
array([0, 1]...)
# get number of support vectors for each class
clf.n_support_
array([1, 1]...)
Examples

SVM: Maximum margin separating hyperplane

SVM-Anova: SVM with univariate feature selection

Plot classification probability

1.4.1.1. Multi-class classification
SVC and NuSVC implement the “one-versus-one” approach for multi-class classification. In total, n_classes * (n_classes - 1) / 2 classifiers are constructed and each one trains data from two classes. To provide a consistent interface with other classifiers, the decision_function_shape option allows to monotonically transform the results of the “one-versus-one” classifiers to a “one-vs-rest” decision function of shape (n_samples, n_classes), which is the default setting of the parameter (default=’ovr’).

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
SVC(decision_function_shape='ovo')
dec = clf.decision_function([[1]])
dec.shape[1] # 6 classes: 4*3/2 = 6
6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes
4
On the other hand, LinearSVC implements “one-vs-the-rest” multi-class strategy, thus training n_classes models.

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
LinearSVC()
dec = lin_clf.decision_function([[1]])
dec.shape[1]
4
See Mathematical formulation for a complete description of the decision function.

Examples

Plot different SVM classifiers in the iris dataset

1.4.1.2. Scores and probabilities
The decision_function method of SVC and NuSVC gives per-class scores for each sample (or a single score per sample in the binary case). When the constructor option probability is set to True, class membership probability estimates (from the methods predict_proba and predict_log_proba) are enabled. In the binary case, the probabilities are calibrated using Platt scaling [9]: logistic regression on the SVM’s scores, fit by an additional cross-validation on the training data. In the multiclass case, this is extended as per [10].

Note

The same probability calibration procedure is available for all estimators via the CalibratedClassifierCV (see Probability calibration). In the case of SVC and NuSVC, this procedure is builtin to libsvm which is used under the hood, so it does not rely on scikit-learn’s CalibratedClassifierCV.

The cross-validation involved in Platt scaling is an expensive operation for large datasets. In addition, the probability estimates may be inconsistent with the scores:

the “argmax” of the scores may not be the argmax of the probabilities

in binary classification, a sample may be labeled by predict as belonging to the positive class even if the output of predict_proba is less than 0.5; and similarly, it could be labeled as negative even if the output of predict_proba is more than 0.5.

Platt’s method is also known to have theoretical issues. If confidence scores are required, but these do not have to be probabilities, then it is advisable to set probability=False and use decision_function instead of predict_proba.

Please note that when decision_function_shape='ovr' and n_classes > 2, unlike decision_function, the predict method does not try to break ties by default. You can set break_ties=True for the output of predict to be the same as np.argmax(clf.decision_function(...), axis=1), otherwise the first class among the tied classes will always be returned; but have in mind that it comes with a computational cost. See SVM Tie Breaking Example for an example on tie breaking.

1.4.1.3. Unbalanced problems
In problems where it is desired to give more importance to certain classes or certain individual samples, the parameters class_weight and sample_weight can be used.

SVC (but not NuSVC) implements the parameter class_weight in the fit method. It’s a dictionary of the form {class_label : value}, where value is a floating point number > 0 that sets the parameter C of class class_label to C * value. The figure below illustrates the decision boundary of an unbalanced problem, with and without weight correction.

../_images/sphx_glr_plot_separating_hyperplane_unbalanced_001.png
SVC, NuSVC, SVR, NuSVR, LinearSVC, LinearSVR and OneClassSVM implement also weights for individual samples in the fit method through the sample_weight parameter. Similar to class_weight, this sets the parameter C for the i-th example to C * sample_weight[i], which will encourage the classifier to get these samples right. The figure below illustrates the effect of sample weighting on the decision boundary. The size of the circles is proportional to the sample weights:

../_images/sphx_glr_plot_weighted_samples_001.png
Examples

SVM: Separating hyperplane for unbalanced classes

SVM: Weighted samples
"""
result = chain.invoke({'text':text})

# print(result)
chain.get_graph().print_ascii()