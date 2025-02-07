# Supplementary Files  

Repository: [GitHub - braindecoding/supl](https://github.com/braindecoding/supl)  

The **Supplementary Files** directory contains two subfolders under the **experiments** folder:  
1. **miawaki** – using the Miyawaki dataset  
2. **vg** – using the Van Gerven dataset  

## Miawaki Folder  

This folder contains experiments using the Miyawaki dataset, including:  
- Reports in **.xlsx** and **.csv** formats  
- **knn** folder: plots of results for each batch, iteration, etc.  

## VG Folder  

This folder contains:  
- Reports in **.csv** and **.xlsx** formats  
- **plot** folder: reports for each iteration, batch, **z** value, and intermediate dimension analysis  

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
