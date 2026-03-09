def dibujoMCS(ce_clasI):
    # ¿Cómo se distribuyen los CE y los MCS en las distintas clases?
    fig = plt.figure(figsize = (10,20))
    ncol,nfil = 6, 2
    igraf=0

    
    ce_clasI['mcs_class']=ce_clasI.mcs_class.cat.reorder_categories(["DSL","DLL","CCC","MCC"],ordered=True)
    ce_clasI['cent_lat']=ce_clasI.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326).y
    ce_clasI['cent_lon']=ce_clasI.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326).x
    ce_clasI['day_yr'] = ce_clasI.time.dt.dayofyear

    
    igraf = igraf + 1
    # Distribución de CEs (para cada tiempo, los elementos son disjuntos y solo los clasifico y cuento)
    ces_clases = ce_clasI.groupby(["mcs_class"], observed=True).mcs_id.count()
    ax = fig.add_subplot(ncol,nfil,igraf)
    ax.bar(ces_clases.index, ces_clases/ces_clases.sum()*100)
    ax.grid()
    ax.set_title('Count of CEs per type. Total='+str(ces_clases.sum()))
    ax.set_ylabel('[%]')

    igraf = igraf + 1
    # Distribución de MCSs (cuento cada identificador mcs_id una sola vez)
    mcs_clases=ce_clasI.groupby(["mcs_class"], observed=True).mcs_id.nunique()
    ax = fig.add_subplot(ncol,nfil,igraf)
    ax.bar(mcs_clases.index, mcs_clases/mcs_clases.sum()*100)
    ax.grid()
    ax.set_title('Count of MCSs per type. Total='+str(mcs_clases.sum()))
    ax.set_ylabel('[%]')
    
    igraf = igraf + 1
    #Localización en latitud típica por CE (por MCS sería muy complicado hacerlo bien)
    ax = fig.add_subplot(ncol,nfil,igraf)
    ce_clasI.boxplot(column='cent_lat',by='mcs_class',ax=ax);
    ax.set_ylabel('Mean latitude');
    ax.set_xlabel('')

    igraf = igraf + 1
    #Localización en longitud típica por CE (por MCS sería muy complicado hacerlo bien)
    ax = fig.add_subplot(ncol,nfil,igraf)
    ce_clasI.boxplot(column='cent_lon',by='mcs_class',ax=ax);
    ax.set_ylabel('Mean longitude');
    ax.set_xlabel('')
    
    igraf = igraf + 1
    #Localización temporal típica por CE (por MCS sería muy complicado hacerlo bien)
    ax = fig.add_subplot(ncol,nfil,igraf)
    ce_clasI.boxplot(column='day_yr',by='mcs_class',ax=ax);
    ax.set_ylabel('Timing in the year');
    ax.set_xlabel('')

    igraf = igraf + 1
    # Boxplot del área de cada MCS clasificada por tipo 
    # (para cada tiempo, sumo el área del mismo MCS)
    copia = ce_clasI.copy()
    copia['str_class']=copia.mcs_class.astype(str)
    mcs_area = copia.groupby(["mcs_id","itime","str_class"]).area_km2.sum().to_frame()
    mcs_subset = mcs_area.reset_index(level=['mcs_id', 'itime'], drop=True)
    mcs_subset = mcs_subset.reset_index()
    mcs_subset['str_class']=pd.Categorical(mcs_subset['str_class'], categories=['DSL','DLL','CCC','MCC'], ordered = True)
    ax = fig.add_subplot(ncol,nfil,igraf)
    mcs_subset.boxplot(column='area_km2',by='str_class',ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel('')
    ax.set_ylabel('Total area [km$^2$]')
    
    igraf = igraf + 1
    #Duración total MCS
    ax = fig.add_subplot(ncol,nfil,igraf)
    n = ce_clasI.mcs_id.nunique()
    gb = ce_clasI.groupby(["mcs_id","mcs_class"], observed=True)
    dur = ((gb.time.max() - gb.time.min()).dt.total_seconds() / 3600).rename("duration_h").dropna().to_frame()
    dur.boxplot(by='mcs_class',ax=ax)
    ax.set_ylabel('Duration [h]')
    ax.set_xlabel('')

    
    igraf = igraf + 1
    # temperatura brillo promedio en cada MCSs clasificado por tipo
    # (para cada tiempo, sumo la precipitación media pesada por el área abarcada)
    copia['tb_area']=copia['area_km2']*copia['mean_tb']
    mcs_tb = copia.groupby(["mcs_id","itime","str_class"]).tb_area.sum().to_frame() 
    mcs_tb['mean_tb']=mcs_tb['tb_area'] / mcs_area['area_km2']
    mcs_tb_subset = mcs_tb.reset_index(level=['mcs_id', 'itime'], drop=True)
    mcs_tb_subset = mcs_tb_subset.reset_index()
    mcs_tb_subset['str_class']=pd.Categorical(mcs_tb_subset['str_class'], categories=['DSL','DLL','CCC','MCC'], ordered = True)
    ax = fig.add_subplot(ncol,nfil,igraf)
    mcs_tb_subset.boxplot(column='mean_tb',by='str_class',ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Mean Tb [K]')
    
    
    igraf = igraf + 1
    # Temperatura de brillo mínima por MCS:
    ax = fig.add_subplot(ncol,nfil,igraf)
    #ce_clasI.boxplot(column='max_tp',by='mcs_class',ax=ax)
    mcs_tb_min = copia.groupby(["mcs_id","itime","str_class"]).min_tb.min().to_frame() 
    mcs_tb_min_subset = mcs_tb_min.reset_index(level=['mcs_id', 'itime'], drop=True)
    mcs_tb_min_subset = mcs_tb_min_subset.reset_index()
    mcs_tb_min_subset['str_class']=pd.Categorical(mcs_tb_min_subset['str_class'], categories=['DSL','DLL','CCC','MCC'], ordered = True)
    mcs_tb_min_subset.boxplot(column='min_tb',by='str_class',ax=ax)
    ax.set_ylabel('Minimum Tb [K]')
    ax.set_xlabel('')

    
    igraf = igraf + 1
    # Precipitación promedio en cada MCSs clasificado por tipo
    # (para cada tiempo, sumo la precipitación media pesada por el área abarcada)
    copia['pcp_area']=copia['area_km2']*copia['mean_tp']
    mcs_pcp = copia.groupby(["mcs_id","itime","str_class"]).pcp_area.sum().to_frame() 
    mcs_pcp['mean_pcp']=mcs_pcp['pcp_area'] / mcs_area['area_km2']
    mcs_pcp_subset = mcs_pcp.reset_index(level=['mcs_id', 'itime'], drop=True)
    mcs_pcp_subset = mcs_pcp_subset.reset_index()
    mcs_pcp_subset['str_class']=pd.Categorical(mcs_pcp_subset['str_class'], categories=['DSL','DLL','CCC','MCC'], ordered = True)
    ax = fig.add_subplot(ncol,nfil,igraf)
    mcs_pcp_subset.boxplot(column='mean_pcp',by='str_class',ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Mean precip [mm·h$^{-1}$]')
    
    
def dibujoCE(ce_trackI):
    fig = plt.figure(figsize = (10,8))
    
    n = ce_trackI.mcs_id.nunique()
    gb = ce_trackI.groupby("mcs_id")
    
    ###################################################################
    # Área máxima cubierta por un MCS en un tiempo dado (usando kernel density estimation)
    ax = fig.add_subplot(2,2,1)
    
    
    # Dist of max area
    x = gb.area_km2.max()
    x.plot.kde(ax=ax, label="CE")

    # Esta parte es para poder tener en cuenta en cada paso de tiempo TODO el área de un mismo MCS porque
    # a veces están a trozos en el mismo tiempo. Esta sería la manera más correcta, entiendo yo.
    mcs = ce_trackI.groupby(["mcs_id", "itime"]).area_km2.sum()
    x2 = mcs.groupby("mcs_id").max()
    x2.plot.kde(ax=ax, label="MCS")
    
    ax.text(0.99, 0.03, f"$N={n}$", size=9, ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlim(xmin=1000); ax.set_xscale("log")
    ax.set_ylim(ymin=0)
    ax.legend(loc="center right")
    ax.set_xlabel("Max area [km$^2$]")
    ax.grid()
    #ax.set_title('Área máxima cubierta por MCS')
    print('Hay que mirar cómo es en observaciones y cómo varía con distintos umbrales para T')
    
    
    ###################################################################
    # Distribución de duración del MCS frente a área máxima cubierta en un tiempo dado
    
    ax = fig.add_subplot(2,2,2)
    
    
    area = gb.area_km2.max()
    dur = ((gb.time.max() - gb.time.min()).dt.total_seconds() / 3600).rename("duration_h")
    data = pd.concat([area, dur], axis="columns")
    
    # Manual KDE, so we can share a colorbar
    kde = gaussian_kde(np.column_stack((data["area_km2"], data["duration_h"])).T)
    X, Y = np.mgrid[0:5e6:100j, 0:30:100j]
    pos = np.column_stack((X.ravel(), Y.ravel())).T
    Z = np.reshape(kde(pos).T, X.shape)
    im = ax.contourf(
        X, Y, np.log10(np.where(Z > 0, Z, np.nan)),
        levels=np.linspace(np.log10(5e-11), np.log10(5e-7), 19),
        extend="max",
        cmap=sns.color_palette("Blues", as_cmap=True),
    )
    ax.text(0.99, 0.03, f"$N={n}$", size=9, ha="right", va="bottom", transform=ax.transAxes)
    ax.set_ylabel('Duration [h]')
    ax.set_xlabel('Máx area [km$^2$]');
    
    ###################################################################
    
    ax = fig.add_subplot(2,2,3)
    tb = gb.min_tb.min().astype(float)  # np.sqrt not supported for `Float64`
    
    tp = gb.mean_tp.mean().astype(float)  # TODO: area-weighted average of CE precip at each time?
    data1 = pd.concat([tb, tp], axis="columns")
    
    
    
    ax.plot(tb, tp, ".", ms=5, mec="none", mfc="0.35", alpha=0.9)
    sns.kdeplot(x="min_tb", y="mean_tp",
        fill=False, color=sns.color_palette()[0], alpha=0.6,
        common_norm=True, clip=(0, None),
        ax=ax, data=data1,
    )
    
    ###################################################################
    
    ax = fig.add_subplot(2,2,4)
    
    
    tpmax = gb.max_tp.max().astype(float)  # TODO: area-weighted average of CE precip at each time?
    data2 = pd.concat([tb, tpmax], axis="columns")
    
    ax.plot(tb, tpmax, ".", ms=5, mec="none", mfc="0.35", alpha=0.9)
    sns.kdeplot(x="min_tb", y="max_tp",
        fill=False, color=sns.color_palette()[0], alpha=0.6,
        common_norm=True, clip=(0, None),
        ax=ax, data=data2,
    )
