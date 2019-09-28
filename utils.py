from external_packages import *

def counts_ratios_weights(y_train,y_test):
    ratios=[]
    weights=[]
    counts=[]
    target_column='to_predict'
    for item in [y_train,y_test]:
        ratios.append(item[target_column].value_counts(normalize=True).values)
        counts.append(item[target_column].value_counts(normalize=False).values)
        weights.append(compute_class_weight('balanced',[0,1],item[target_column].values.flatten()))
    weights_df=pd.DataFrame(weights,index=['train_weights','test_weights']).T
    ratios_df=pd.DataFrame(ratios,index=['train_ratios','test_ratios']).T
    counts_df=pd.DataFrame(counts,index=['train_counts','test_counts']).T
    return(pd.concat([counts_df,ratios_df,weights_df],axis=1))

def check_file(filename,location=cwd):    
    
    return os.path.exists(os.path.join(location,filename)),os.path.join(location,filename)

def check_folder(foldername,location=cwd):    
    
    return os.path.exists(os.path.join(location,foldername))

def create_folders(folders):
    for folder in folders:
        if(check_folder(folder)):
            pass
        else:
            print('*** Creating new folder named: ',folder) 
            os.mkdir(folder)

def download_files(files,download_folder):
    for file in files:
        [[_,location]]=file.items()
        file_name=os.path.basename(location)
        exists,_=check_file(file_name,download_folder)
        if(exists):
#             print(file_name,"already exists")
            pass
        else:
            print('*** Downloading : ',file_name)
            try:
                r = requests.get(location, auth=('usrname', 'password'), verify=False,stream=True)
                r.raw.decode_content = True   
                with open(os.path.join(download_folder,file_name), 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
            except:
                raise Exception('Failed')
                
def split_dataset(in_features,fracs,random_state=0):
    datasets=[]

    for frac in fracs:
        temp=in_features.sample(frac=frac, replace=False,random_state=random_state)
        datasets.append(temp)
    return([data for data in datasets])


def get_info(target_set):
    """
    Extracts ['column','Missing','Duplicated','Unique','Type'] information about a pandas dataframe
    """
    info=pd.DataFrame([[x,pd.isna(target_set[x]).sum(),target_set[x].duplicated().sum(),
                        len(target_set[x].unique()),target_set[x].dtype] for x in target_set.columns])
    info.columns=['column','Missing','Duplicated','Unique','Type']
    return(info)

def replace_values_new(x,df,columns):
    def subset(x,df,column):
        type(df[column][0])(x[column])
        subset_df=df[df[column]==x[column]]
        return(subset_df)
    for column in columns:
        temp=subset(x,df,column)
    x.InteractionTime=temp.InteractionTime.max() if (not np.isnan(temp.InteractionTime.mean())) else np.NAN
    return(x)

def scree_plot(pcas,titles=[],figsize=(10,5)):

    fig, main_axes=plt.subplots(len(pcas)//3,3,figsize=figsize)
    main_axes=main_axes.ravel()
    for i,pca in enumerate(pcas):
        main_axes[i].plot(np.cumsum(pca.explained_variance_ratio_))
        main_axes[i].plot(pca.explained_variance_ratio_,'--')
        main_axes[i].set_xlabel('Components')
        main_axes[i].set_ylabel('Explained Variance')
        main_axes[i].set_title(titles[i])
        main_axes[i].legend(labels=['cumulitive sum','ratio']);
        main_axes[i].grid()
    fig,axes=plt.subplots(1,1,figsize=(15,5))
    l=[]
    for i,pca in enumerate(pcas):
        temp,=axes.plot(np.cumsum(pca.explained_variance_ratio_))
        l.append(temp)
    axes.grid()
    axes.set_ylabel('Explained Variance')
    axes.set_xlabel('Components')
    fig.legend([item for item in l],titles,loc="center") 


def biplot(score,actual,coeff,pcax,pcay,labels=None,proj_features=False):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    fig  = plt.figure(figsize=(10,5))
    if isinstance(actual, np.ndarray):
        N=len(np.unique(actual.reshape(-1))) 
        actual=actual.reshape(-1)
    if isinstance(actual, pd.DataFrame):
        N=len(np.unique(actual.values))
        actual=actual.values.reshape(-1)
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    scat=plt.scatter(xs*scalex,ys*scaley,c=actual,cmap=cmap,norm=norm)
    if(proj_features):
        for i in range(n):
            plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5) 
            if labels is None:
                plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
            else:
                plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Classes')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()
    
def plot_cm(actual,predicted,cm_count=True):
    cm=ConfusionMatrix(actual_vector=actual, predict_vector=predicted)
    fig, ax = plt.subplots(figsize=(5,5))
    matrix=pd.DataFrame.from_dict(cm.table,orient='index')
    if(cm_count):
        annot=matrix.values
    else:
        annot=pd.DataFrame.from_dict(cm.normalized_matrix,orient='index').apply(lambda x:round(x,3)).values
    sns.heatmap(matrix,annot=annot,ax=ax,fmt='')
    ax.set(xlabel='Predicted', ylabel='Actual')
    return(cm)

def get_sorted_pc(principalDf,n,to_plot=20,show_main=False,return_data=False):
    temp=principalDf.sort_values(by='PC'+str(n), ascending=False, axis=1).loc['PC'+str(n),:]
    x=pd.DataFrame(temp).head(to_plot)
    if (not show_main and not return_data):
        x.plot.bar(figsize=(20,5))
        plt.title(str(to_plot)+' Most Positive')
        plt.show()
        x=pd.DataFrame(temp).tail(to_plot)[::-1]
        x.plot.bar(figsize=(20,5))
        plt.title(str(to_plot)+' Most Negative')
        plt.show()
        temp.plot.bar(figsize=(20,5))
        plt.title('Overall')
        plt.show()
        return(temp)
    elif(not return_data):
        temp.plot.bar(figsize=(20,5))
        plt.title('Overall')
        plt.show()
        return(None)
    if(return_data):
        return(temp.head(5).append(temp.tail(5)))

class experiment:
    def __init__(self,):
        self.cms_dict={}
        self.cms_list=[]
        self.model=None
        self.train_test_splits=None
        self.models={}
        self.models_trained={}
        self._modelNames=list(self.cms_dict.keys())
        self.cmap_name='hsv'
        self.figsize=(10,7)
        self.predicted=[]
        self.actual=[]
        
        
    def _plot_cm(self,actual,predicted,cm_count=True):
        cm=ConfusionMatrix(actual_vector=actual, predict_vector=predicted)
        fig, ax = plt.subplots(figsize=(5,5))
        matrix=pd.DataFrame.from_dict(cm.table,orient='index')
        if(cm_count):
            annot=matrix.values
        else:
            annot=pd.DataFrame.from_dict(cm.normalized_matrix,orient='index').apply(lambda x:round(x,3)).values
        annot_kws = {"ha": 'left',"va": 'top'}
        sns.heatmap(matrix,annot=annot,ax=ax,fmt='',annot_kws=annot_kws)
        ax.set(xlabel='Predicted', ylabel='Actual')
        return(cm)

    def _plot_cms(self,cms):
        negative=pd.DataFrame()
        positive=pd.DataFrame()
        if(cms):
            for cm in cms:
                temp_n=pd.DataFrame([cm.table[0]])
                negative=pd.concat([negative,temp_n])

                temp_p=pd.DataFrame([cm.table[1]])
                positive=pd.concat([positive,temp_p])

            negative.reset_index(drop=True,inplace=True)
            negative.columns=['True Negative','False Positive']
            positive.reset_index(drop=True,inplace=True)
            positive.columns=['False Negative','True Positive']
            order=['True Negative','False Positive','False Negative','True Positive']
            C_M=positive.join(negative).loc[:,order]
            neg_count=negative.apply(lambda x:sum(x),axis=1)[0]
            pos_count=positive.apply(lambda x:sum(x),axis=1)[0]
            counts=[neg_count,neg_count,pos_count,pos_count]
            sec_index=np.array([])
            colors=[]
            for i in range(len(list(self.cms_dict.keys()))):
                n_times=len(self.train_test_splits)
                sec_index=np.append(sec_index,np.repeat(list(self.cms_dict.keys())[i],n_times))
                colors.append(np.repeat(i,n_times))
            pri_index=list(C_M.index)
            C_M.index=[pri_index,sec_index]
            c=np.array(colors).reshape(-1)
            cmap = plt.get_cmap(self.cmap_name)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=len(list(self.cms_dict.keys())))
            axes=C_M.plot.bar(y=C_M.columns,subplots=True,figsize=self.figsize,layout=(2, 2),sharex=False,legend=False)
            trash=[ax.set_xlabel(ax.title.__dict__['_text']) for ax in axes.reshape(-1)]
            trash=[ax.title.set_text(None) for ax in axes.reshape(-1)]
            for ax in axes.reshape(-1): ax.xaxis.labelpad=10
            for i,ax in enumerate(axes.reshape(-1)):
                for j,p in enumerate(ax.patches):
                    p.set_facecolor(cmap(norm(c))[j])
                    ax.text(p.get_x()+(p.get_width()/4), p.get_height()-p.get_height()/10, str(p.get_height()))
#                     ax.annotate(round(p.get_height()/counts[i],3),(p.get_x()+(p.get_width()/6),p.get_height()/2))
            plt.subplots_adjust(hspace=1.3,wspace=0.5)
    def run_model(self,model,name=None):
        self.model=model
        if(not name):
            self.models[self.model.__class__.__name__]=model
        else:
            self.models[name]=model
        cms=[]
        for i,(X,y,X_,y_) in enumerate(self.train_test_splits):
            self.model.fit(X,y)
            predicted=self.model.predict(X_)
            cm=ConfusionMatrix(y_.values.flatten(), predicted)
            self.actual.append(y_.values.flatten())
            self.predicted.append(predicted)
            cms.append(cm)
        self.cms_dict[self.model.__class__.__name__]=cms
        self.cms_list.append(cms)
        if(not name):
            self.models_trained[self.model.__class__.__name__]=model
        else:
            self.models_trained[name]=model
        
    def plot_comp(self,cmap_name='hsv',figsize=(10,7)):
        self.figsize=figsize
        self.cmap_name=cmap_name
        temp=self._plot_cms(list(np.array([cms for k,cms in self.cms_dict.items()]).flatten()))
        

        
data_folder=os.path.join(cwd,'data')
download_folder=os.path.join(cwd,'download')
processed_data_folder=os.path.join(cwd,'data','processed')
original_data_folder=os.path.join(cwd,'data','original')
cleaned_data_folder=os.path.join(cwd,'data','cleaned')
create_folders([data_folder,download_folder,processed_data_folder,cleaned_data_folder])
main_paths={'data_folder':data_folder,
            'download_folder':download_folder,
            'processed_data_folder':processed_data_folder,
            'original_data_folder':original_data_folder,
            'cleaned_data_folder':cleaned_data_folder}

dataset={'dataset':"https://github.com/rouzbeh-afrasiabi/PublicDatasets/raw/master/stackset.zip"}

toDownload=[dataset]
download_files(toDownload,download_folder)

exists,_=check_file("stackset.csv",original_data_folder)

if(not exists):
    zip_file = zipfile.ZipFile(os.path.join(download_folder,"stackset.zip"), 'r')
    zip_file.extractall(original_data_folder)
    zip_file.close()
dataset_loc=os.path.join(original_data_folder,"stackset.csv")
# final_stackset=final_stackset.groupby(['ID','to_predict']).apply(lambda x: x.sample(1)).reset_index(drop=True)