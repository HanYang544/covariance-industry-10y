LW_WINDOWS=[20,60,120,252]
HALVES=[(21,63),(63,125),(125,250),(250,500),(500,1000)]
EXPERTS=[f'LW-{w}' for w in LW_WINDOWS]+[f'IEWMA-{v}-{c}' for v,c in HALVES]+['Diagonal-IEWMA-21']
