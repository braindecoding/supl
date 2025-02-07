# Supplementary Files  

Code Repository: [GitHub - braindecoding/supl](https://github.com/braindecoding/oaa)  
Code Page: [Page - braindecoding/oaa](https://braindecoding.github.io/oaa/)  
Supplementary Files Repository: [GitHub - braindecoding/supl](https://github.com/braindecoding/supl)  
Supplementary Files Page: [Page - braindecoding/supl](https://braindecoding.github.io/supl/)  

The **Supplementary Files** directory contains two subfolders under the **experiments** folder:  
1. **vg** – using the Van Gerven dataset  
2. **miawaki** – using the Miyawaki dataset  



## VG Folder  

This folder contains:  
- Reports in **.csv** and **.xlsx** formats  
- **plot** folder: reports for each iteration, batch, **z** value, and intermediate dimension analysis  

Running **runvg.bat** inside the **plot** folder generates **FID_Result.csv**, along with calculations and reconstructed images for each latent variable, intermediate dimension, batch size, and iteration.  

## Structure of the **plot** Folder  

After running **runvg.bat**, the **plot** folder contains multiple subfolders, each representing different configurations of latent variables, intermediate dimensions, batch sizes, and iterations.  

### Example Folder Structure:  
```
plot/
│── 9_128_10_500/
│── 9_128_10_1000/
│── 9_128_10_1500/
│   ├── plot/
│   │   ├── fig.png
│   │   ├── graph.png
│   │   ├── result.png
│   ├── rec/
│   │   ├── image_0.png
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   ├── image_3.png
│   │   ├── image_4.png
│   │   ├── image_5.png
│   │   ├── image_6.png
│   │   ├── image_7.png
│   │   ├── image_8.png
│   │   ├── image_9.png
│   ├── score/
│   │   ├── score.csv
│   ├── stim/
│   │   ├── image_0.png
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   ├── image_3.png
│   │   ├── image_4.png
│   │   ├── image_5.png
│   │   ├── image_6.png
│   │   ├── image_7.png
│   │   ├── image_8.png
│   │   ├── image_9.png
```

### Folder Details:  
- **plot/**: Contains visualization results, including:  
  - `fig.png`: A summary figure.  
  - `graph.png`: A graphical representation of results.  
  - `result.png`: The final processed result image.  

- **rec/**: Stores reconstructed images (`image_0.png` to `image_9.png`).  

- **score/**: Contains `score.csv`, which holds the evaluation scores for the generated images.  

- **stim/**: Stores the original stimulus images corresponding to the reconstructed ones.  

This structure is repeated for each configuration folder (e.g., `9_128_10_500`, `9_128_10_1000`, `9_128_10_1500`), where the numbers indicate different parameter settings.  

## Understanding the Folder Naming Convention  

Each folder inside **plot/** follows this naming format:  
```
K_intermediateDim_batchSize_maxIter/
```
Where:  
- **K** → Number of latent variables (e.g., `9`)  
- **intermediateDim** → Intermediate dimension of the model (e.g., `128`)  
- **batchSize** → Batch size during training (e.g., `10`)  
- **maxIter** → Maximum number of iterations (e.g., `500`, `1000`, `1500`)  

## How the **fidvg.py** Script Works  

The **fidvg.py** script is responsible for computing the **Frechet Inception Distance (FID)** score and organizing the results:  

1. The script receives four parameters when executed:  
   ```bash
   python fidvg.py K intermediate_dim batch_size maxiter
   ```
2. It generates the folder name based on the parameters:  
   ```python
   rootfolder = f"{K}_{intermediate_dim}_{batch_size}_{maxiter}/"
   ```
3. Inside this folder, it creates:
   - **stim/** → Stores original stimulus images
   - **rec/** → Stores reconstructed images  
4. It calculates the **FID score** by comparing images in **stim/** and **rec/**.  
5. The FID score is saved in **FID_Results.csv** with the following format:  
   ```csv
   K, intermediate_dim, batch_size, maxIter, fid_value
   ```

### Example Execution:  
If the script is run with:  
```bash
python fidvg.py 9 128 10 1000
```
It will generate a folder:  
```
plot/
│── 9_128_10_1000/
│   ├── plot/
│   ├── rec/
│   ├── score/
│   ├── stim/
```
And append a line to **FID_Results.csv**:  
```
9,128,10,1000, <FID Score>
```

## Miawaki Folder

To reduce the size of the supplementary file, the Miyawaki folder contains several sampled results. This folder includes experiments using the Miyawaki dataset, such as:

- Reports in **.xlsx** and **.csv** formats
- **knn** folder: plots of results for each batch, iteration, etc.

### Structure of the **Miawaki** Folder

```
miawaki/
│── knn/
│   ├── 18_512_40_1500_2.png
│   ├── 18_512_50_500_1.png
│   ├── 18_512_50_500_2.png
│   ├── 18_512_50_1000_1.png
│   ├── 18_512_50_1000_2.png
│   ├── 18_512_50_1500_1.png
│   ├── 18_512_50_1500_2.png
├── FID_Results_Figure.png
├── FID_Results_My.csv
├── FID_Results_Vg.csv
├── FID_Results18.csv
├── FID_Results18.xlsx
├── FID_Results512-1300.csv
├── FID_ResultsKNNmiya.csv
├── FID_ResultsKNNmiya.xlsx
```

### Folder Details:

- **knn/**: Contains images representing results from different experimental settings. The naming convention follows:

  - `18_512_50_500_1.png` → 18 latent variables, 512 intermediate dimensions, batch size 50, 500 iterations (first sample)

- Stores result reports, including:

  - `FID_Results_Figure.png`: Visualization of FID scores.
  - `FID_Results_My.csv`: FID results for Miyawaki dataset.
  - `FID_Results_Vg.csv`: FID results for Van Gerven dataset.
  - `FID_Results18.csv/xlsx`: General results.
  - `FID_ResultsKNNmiya.csv/xlsx`: KNN-based FID results for Miyawaki dataset.

