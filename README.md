# -C--Users-ASUS-CMS_Analysis-CMS_Analysis_Macroedtech.py-
This research is concerned with the problem of Higgs and non-Higgs events classification using machine learning algorithms. The source of the dataset lies in simulations analogous to those performed at CERN. This study aims to explore how computer models could help detect certain patterns associated with the Higgs boson.
## Dataset Description
Among the entries, over fifteen thousand detail particle crashes. Every instance includes multiple characteristics rooted in physical principles. Energy appears measured in GeV. Motion along three axes - x, y, z - is recorded separately. Invariant mass shows up in similar units. Detectors contribute readings tied to signal strength. When behavior matches expectations for the Higgs boson, is_higgs equals one. All other cases assign it zero.

## Methodology
Beginning at the earliest stage, data cleansing forms the foundation - eliminating irrelevant entries like identification tags while checking thoroughly for gaps within records. Following that phase, interpretation improved when a derived feature entered the set, aiding clarity in particle behavior analysis.
For every particle, during the process of generating features, its overall momentum in GeV emerged from combining contributions across all spatial directions. From Px squared to Py and onward to Pz, each term added in quadrature shaped the result. The final value took form by extracting the root after summing these components. Through this method, magnitude arose without directional bias. Calculation followed a consistent mathematical path regardless of individual direction weights.

## Machine Learning Models
To identify event classes, multiple machine learning methods were used. Logistic regression formed the starting point, revealing straightforward associations across variables. Complex interactions found better expression through random forest, chosen as the central approach. Alongside it, support vector machines offered another angle, contrasting with simpler neighbor-based predictions from KNN. Each method contributed a separate view into how data points align within feature space.

## Model Evaluation
Among the criteria applied was accuracy, followed by precision, then recall, alongside the F1 score. Examination of class separation - Higgs versus non-Higgs - drew from confusion matrices generated per algorithm.
Despite moderate performance, logistic regression was outperformed by both SVM and KNN, which showed strong accuracy. Non-linear patterns were effectively captured by random forest, leading to superior outcomes across evaluations.

## Key Insights
Among all variables tested, invariant mass carried the greatest influence. Approaching 125 GeV increases likelihood of Higgs classification. Notably, Random Forest outperformed Linear Regression - non-linearity in the task favored its structure. Clear separation between classes emerged, partly because synthetic data shaped distinct boundaries. Performance gaps arose naturally under such controlled conditions.

## Tools and Technologies
This work relied on Python, along with supporting tools. Data handling drew from Pandas alongside NumPy. Visual outputs emerged through Matplotlib combined with Seaborn. Machine learning procedures applied Scikit-learn as the base framework.

## Conclusion
This work demonstrates an effective application of machine learning within particle physics. Although simulation-generated data was used, insight into practical solutions emerges through the model's behavior. A clear path forms when methods adapt to complex patterns found in such environments. The approach shows potential without relying on real-world inputs at this stage. Understanding grows as structures in data become visible via computational analysis.

## Organization
MacroEdtech.
## Supervisor
Sagar Sakalley

Founder of MacroEdTech
## Project Created By
Atufa Vora

Educational Content Writer, Physics
MacroEdtech