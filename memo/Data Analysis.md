# Pandas
## Functions
### Read dssata
- `read_table` or `read_csv`

### Merge data
[Pandas: Merge, join and concatenate](https://pandas.pydata.org/pandas-docs/stable/merging.html)

- **pd.merge()**
    ```
    pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
             left_index=False, right_index=False, sort=True,
             suffixes=('_x', '_y'), copy=True, indicator=False,
             validate=None)
    ```
    Common set args: `how`, `on`
    - `how`: `left`, `right`, `outer` or `inner`. 决定融合的column组合的标准，以左、右为、并集或者交集。
    - `on`: Columns (names) to join on, must be found in both the left and right. 需要已哪几列作为融合的基础列。

### Analysis
- **pd.pivot_table()**
    ```
    pandas.pivot_table(data, values=None, index=None, columns=None, 
                       aggfunc='mean', fill_value=None, margins=False, 
                       dropna=True, margins_name='All')
    ```
    Common set args: `values`, `index`, `columns` and `aggfunc`
    - `values`: column to aggregate. 需要计算的值所在的列
    - `index`:  Keys to group by on the pivot table index. 透视表index(row)属性
    - `columns`:  Keys to group by on the pivot table column. 透视表列属性
    - `aggfunc`: 对`values`值的计算法方法
    
    ***References***
    - [pandas API: pandas.pivot_table](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html)