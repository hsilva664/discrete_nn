# About

Contains code from the MaskedNNRegression benchmark. For information on the command-line arguments, check the main parser in the `aux` directory, as well as the per-method additional parser, located inside each method.

Some examples of running `main.py`:

```bash
# CP
python3 main.py --out_id AAx --out_h5 df.h5 --method cp --optim_str sgd --lr 0.01 --seed 5

# ARMS + Escort
python3 main.py --out_id pOz --out_h5 df.h5 --method arms --mask_f escort --optim_str rmsprop --lr 0.01 --seed 5 --n 10
```

In both cases, output will be saved in different tables inside the file `out/df.h5`. Use HDFView or Pandas to load the file and inspect the contents.

There are two tables:
- `params`: column `keys` contains the arguments and column `values` contains their values
- `runs`: contains the losses logged throughout training

To load them on Pandas, you can, for example, do:

```python
df = (pd.read_hdf("out/df.h5", "runs")  
      .melt(ignore_index=False).reset_index()  
      .pivot(index="Epochs", columns=["Id", "Quantity"], values="value"))  
 
id_df = (pd.read_hdf("out/df.h5", "params")  
          .assign(keys=lambda d: d['keys'].map(lambda v: eval(v)) )  
          .assign(values=lambda d: d['values'].map(lambda v: eval(v)))
           .explode(["keys", "values"])
           .pivot(columns="keys", values="values")
         )
```
