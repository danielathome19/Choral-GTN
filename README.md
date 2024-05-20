# About
Choral-GTN is a deep learning system (comprised of a hybrid Generative Transformer Network architecture, Music Generator Callback Interface, and Rule-based Post-Processing Script) trained to generate realistic four-part (SATB) classical choral music. The system also provides a new dataset, the **Choral Harmony Optimized Repository for AI Learning (CHORAL) Database**, that provides the four isolated voice parts and their concatenation for 1000 unique pieces of classical choral music.

To find out more, check out the provided research paper:
  * **Doctoral Dissertation**: "Choral Music Generation: A Deep Hybrid Learning Approach" (DOI: [10.13140/RG.2.2.10418.62400](http://doi.org/10.13140/RG.2.2.10418.62400)) 
  * **Journal Paper**: "TBD" (DOI: [TBD](#))
  * Also contained in the ["PaperAndPresentation"](https://github.com/danielathome19/Choral-GTN/tree/master/PaperAndPresentation) folder is the dissertation paper, journal paper (and supplement), and presentation of the research.
  * The dissertation defense can be watched at https://youtu.be/Rk9koF0V-2M.

# Usage
See:
  * TBD for a live demo of the full prediction system.
  * https://github.com/danielathome19/Choral-GTN/releases for a downloadable local demo of the prediction system.

For data used in my experiments:
  * All datasets can be found in **Data**.
  * My most recent pre-trained weights can be found in **Weights**.

**NOTE:** these folders should be placed in the **same** folder as "main.py". For folder existing conflicts, simply merge the directories.

In main.py, the "main" function acts as the controller for the model, where calls to train the model, create a prediction, and all other functions are called. One may also call these functions from an external script ("from main import XYZ", etc.).

To choose an operation or series of operations for the model to perform, simply edit the main function before running. Examples of all function calls can be seen commented out within main.

# Contribution
TBD

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.
  
# License
The code is licensed under CC0 License 1.0.

The database was compiled from free and open sources with respect to the original file creators and sequencers. This work is purely for educational and research purposes, and no copyright is claimed on any files contained within the database.

# Citation
If you use this code for your research, please cite this project as either the *dissertation* (**Choral Music Generation: A Deep Hybrid Learning Approach**):
```
@software{Szelogowski_Choral-GTN_2024,
 author = {Szelogowski, Daniel},
 doi = {10.13140/RG.2.2.10418.62400},
 month = {March},
 title = {{Form-NN}},
 license = {CC0-1.0},
 url = {https://github.com/danielathome19/Choral-GTN},
 version = {1.0.0},
 year = {2024}
}
```
or the *journal paper* ():
```

```
or the *dataset* (**Choral Harmony Optimized Repository for AI Learning (CHORAL) Database**):
```
@misc{Szelogowski_CHORAL-Dataset-And-Choral-GTN_2024,
 author = {Szelogowski, Daniel},
 doi = {10.13140/RG.2.2.10418.62400},
 month = {March},
 title = {{CHORAL-Database-And-Choral-GTN}},
 license = {CC0-1.0},
 url = {https://github.com/danielathome19/Choral-GTN},
 year = {2024}
}
```
