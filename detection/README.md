# Instalēšanas instrukcija

1. Uzinstalējiet Anaconda vai Miniconda;
2. Izveidojiet virtuālo conda vidi:
    * Atvēriet Anaconda prompt komandvidi (Windows) vai termināli (MacOs, Linux);
    * Palaidiet komandu:
    ``` bash
    conda env create -f detection.yaml
    ```
    Conda automātiski izveidots virtuālo vidi *detection* un uzinstalēs visas nepieciešamas bibliotēkas.

# Datu sagatavošana un modeļa apmācība
Secīgi atvēriet *Jupyter Notebook (ipynb)* datnes un, vadoties pēc komentāriem, palaidiet kodu tajos.

# Lietotnes palaišana
1. Atgriezieties uz Anaconda prompt komandvidi vai termināli un aktivizējiet conda vidi, palaižot komandu:
``` bash
conda activate detection
```
2. Komandvidē vai terminālī ar aktivizēto conda vidi atvēriet šo mapi un palaidiet komandu:
``` bash
streamlit run app.py
```
3. Lietotnes darbības pārtraukšanai komandvidē vai terminālī nospiediet Ctrl+C vai aizvēriet komandvidi vai termināli.