import pandas as pd
import numpy as np
import re
import difflib
import logging

from pandas import DataFrame, Series
from typing import List, Literal, Any
from IPython.display import display
from copy import copy
from csv import Sniffer

# ? use pydantic for data validation?

class VarMap():
    
    def __init__(self,
                 df: DataFrame,
                 vintage:str,
                 varid:str,
                 desc:str, 
                 *,
                 values:str|DataFrame|None=None, # if multiple values for variable use dataframe
                 hiermethod:Literal["auto","leading","delimited"]="auto",
                 hiercol:str|None=None,
                 pattern:str|None=None,
                 orderby:str|None=None):
        
        
        # ? interpret varid column and store as meta field? (_x) e.g. hierarchical delimited
        # ! if pattern then hiercol must be provided
            # ? I think hiercol can be on its own as an indicator for the standard parsers?
            # ? what if there are multiple columns that indicate hierarchy in an already parsed form?        
        self._vintage = vintage
        self._hiercol = hiercol
        # infer hierarchies and pass pattern
        # df["Hierarchies"] = df.join(self.infer_hierarchy(self,pattern,inplace=True))
        
        # transform dataframe to standardized format and store
            # rename value column to map_index
            # if order by is provided, that column is used to sort, otherwise assumes that default order is meaningful
                # if orderby column contains a delimited hierarchy, can't use standard sort
        # index by row and id
        #df.reset_index().set_index(varid,append=True)[desc]
        # ! need to error handle in case the var id is already an index?
        
        # ? allow adding other fields that might have useful data besides description?
        
        self.df = df        
        self.df = self.df.rename(columns={varid:"id",desc:"description"})        
        self.df = self.infer_hierarchy(hiermethod,hiercol=hiercol,pattern=pattern,inplace=True)


    def _repr_html_(self):
        # ? build general style method?
        return self.df.style.set_properties(**{"text-align":"left","white-space":"pre-wrap"})._repr_html_()
            
        
    def __repr__(self):                
        return repr(self.df)

    # TODO: add ordering based on vintage?
        
    @staticmethod
    def from_csv(path:str,varid,desc,vintage,**kwargs) -> 'VarMap':
        df = pd.read_csv(path,**kwargs)
        return VarMap(df,vintage,varid,desc)        
    
    @staticmethod 
    def from_json(path:str) -> "VarMap": # TODO
        
        pass
    
    @staticmethod
    def from_xlsx(path:str) -> "VarMap": # TODO
        
        pass
    
    @staticmethod
    def from_xml(path:str) -> "VarMap": # TODO
        
        pass

    @property
    def vintage(self):
        return self._vintage
    
    @vintage.setter
    def set_vintage(self,vintage:str):
        self._vintage=vintage
    
    def add_values(self,df:DataFrame):
        # NOTE id's may be given as column names rather than rows
    
        pass

    # ? use overloading for default vs user-defined and for providing function?
    def infer_hierarchy(self, 
                        hiermethod:Literal["auto","leading","delimited","regex"]="auto",                           
                        *,
                        hiercol:Literal["id","description"]|None=None, 
                        pattern:str|None=None,                          
                        inplace:bool=False) -> "VarMap": # ? can return None?
        # ? should this have an inplace parameter?
            # ? should default be true since it would only ever be called to fix a bad auto infer?
            # maybe remove inplace and always return a VarMap then the change created can be seen
        # ? should method have a default? since it's called by auto on creation is there a need to default it when manually called?
            # ? only if not done on create so that it can be manually applied - what's more user-intuitive?
        # ! TODO: regex
        # ! TODO: allow user to pass a function to identify level
            # user-defined is assumed to return the level all others return indicators that need to be reduced
        # TODO: json inference from nesting
        
        
                                    
        def leading(rowdesc:str) -> int:                
            return len(rowdesc) - len(rowdesc.lstrip(pattern))
                        
        def delimited(rowdesc:str) -> int:            
            return len(rowdesc.split(pattern))-1
        
        
        def pattern_sniff(method:function, s:Series) -> str:
            # ? maybe we can cache so it can be used in apply
                                                        
            if method==delimited:
                # find non-space non-alnums used to delimit
                s1 = s.apply(lambda x:Sniffer().sniff(x.replace(" ","")).delimiter)                
                
            elif method==leading:                                
                # find non-alnums used to potentially lead
                s1 = s[~s.str[0].isalnum()].str[0]                                
            
            # find the most common pattern
            pattern = s1.groupby(s1).count().sort_values(ascending=False).index[0]
            return pattern

            
        def simplify(indicators:Series) -> Series:
            icnts = np.sort(indicators.unique())
            return icnts[1]-icnts[0]

        
        def vectorized_indices(lvls:list) -> Series:                                                    
            # create vectors to represent nested indices            
            vectors = []

            # based on current level, increment the relevant list element for the previous vector            
            for i,lvl in enumerate(lvls):    
                try:                    
                    new_vlist = [int(v) for v in vectors[i-1].split(".")][:lvl]
                except: # occurs on index 0
                    new_vlist = []
                                
                new_len = len(new_vlist)
                if new_len == lvl: # increment existing level
                    new_vlist[lvl-1] += 1                                
                else: # add sub-level (if lvl jumps, fill with 0's)
                    new_vlist = new_vlist + ([0]*(lvl - new_len - 1)) + [1]                    
                        
                vectors.append(".".join([str(v) for v in new_vlist]))

            return pd.Series(vectors)

        
        def validate():
            # ! need to validate hierarchies
            # how to identify a possibly valid hierarchy
            # it has valid changes in nesting levels
            # a nested level occurs a non-trivial number of times        
            # ? option to fail or warn on invalid?
            if not np.array_equal(df["leading"].unique(),icnts):
            # TODO: provide specific row where error occurs
                # for the indent that appeared out of order, find the row where it first occurs
                raise IndentationError("Data order generated an invalid hierarchy")
            
            # ! df.level.drop_duplicates().is_monotonic_increasing
            
            pass
        
        
        df = self.df.copy()
        
        if hiermethod=="auto":
            # ? iterate methods, and if generates a valid hier then use and skip the rest?
                # ? run all and then warning if successful methods generate inconsistent
            # run all, validate all - amongst valid prefer in order of delimited id, leading desc, delimited desc
            
                        
            # auto (sniff) delimited id
                # should auto delimited sniff the description? i.e. solve the us census delimiter
    
            # auto (sniff) leading desc - strip leading characters if the leading character is not alphanumeric
 
            
            # ! what if method is auto but they also included a pattern?
                # set as default and warning that it was ignored?
            
            
            pass
        else:
            if hiercol is None or pattern is None:
                raise TypeError("If either hiercol or pattern is given then both are required")            
            elif hiermethod=="leading":
                func = leading
            elif hiermethod=="delimited":
                func = delimited
            elif hiermethod=="regex":
                
                pass
                        
        df["level"] = df[hiercol].fillna("").map(func)
        df["level"] = df["level"] / df[hiercol].fillna("").agg(simplify)
        
        # validate?
        
        df["vector"] = vectorized_indices(df["level"].to_list()) 
        
        # df["nested_description"] = 
        
        if inplace:            
            self.df = df
            return None            
        else:            
            vm = copy(self)
            return vm       
    

class XVarMap():
    
    def __init__(self, varmaps:List[VarMap],refframe):
        
        self._refframe = refframe
        
        # join varmaps on id and index by refframe's row
        
        
        # full data
            # grouped columns by id, description, depth, summary and then granular by vintage
            # can run versions of the full data that only show certain groups
            
        # maybe streamlit can help?
            
        # each cell stores the id, description, and depth
        
        pass
        
    def save(self,path:str):
        
        pass

    @staticmethod
    def load(path:str) -> 'XVarMap':
        
        pass
        
    def set_refframe(self,vintage:str):
        """Uses the provided VarMap id key to designate that VarMap's variables as the frame of reference"""
        # on changing refframe need to resort by row number of the new frame of reference to privilege the refframe's ordering
        pass
    
    def refframe_dir(self,vintage:str):
        # find whether vintage is before or after refframe
        # and/or give all cascaded impact frames
        pass
    
    def validate_ids(self):
        # check for uniqueness of 
        pass
    
    def validate_transform(self,vintage:str) -> None:
        
        if vintage == self._refframe:
            raise ValueError("'vintage' cannot be the frame of reference. This transform is done relative to the frame of reference")
    
    def distance(self,vintage:str):
        # edit distance for vintage's description vs refframe's description
        # all row distances vs avg vs count
        
        pass
    
    def hiererrors(self,vintage:str):
        # identify and summarize hierarchy errors in vintage vs refframe
        # all row errors vs avg vs count
        # ? combine distance and hiererrors into a single analysis?
        pass
    
    def delta(self):
        # change in distance and hierarchy errors
        
        pass
    
    def find_break(self):
        # identify row where subsequent errors increase significantly (how to define?)
        # ? need a way to annotate rows as accepted and store notes to skip so we can find the next break?
        pass            
    
    def show_table(mode:Literal["values"],post_transform:bool=False):
        # show interactive table
        
        pass
    
    
    def validateop(self):
        # test if a transformation will collide with an existing mapping and error out if so
        
        # test if operation targets the frame of reference
        pass
    
    def apply_notes(self):
        
        # make a note of remapped, accepted
        
        pass
    
    def accept(self):
        
        # accept divergence
        
        pass
    
    def cascade(self, mode:Literal["relative","subset","upstream","downstream"]="relative", subset:List[Any]|None=None):
        
        if mode=="subset" and not subset:             
            raise TypeError("A list of vintages must be provided")
            
        
        # ? if upstream/downstream overlaps _refframe, should it continue past?
        # if row # matches but not the variable id then need to confirm if cascade
        
        pass
        
    def autoadjust(self):
        # smart correction
        # join then run the auto-correct function if bool
        #"""Basic outer join using variable id's."""
        # # ? Can you join any or is it always joined against the frame of reference?
        #     # maybe xstitch is to join against non-frame of reference VarMaps
        
        pass
    
    
    def shift(self,vintage:str,from_row:int|None,by_nrows:int,
              inplace:bool=False,impact_summary:bool=True):
        """Modifies the XVarMap by shifting a VarMap's variables up or down a number of rows against the frame of reference variables"""
        # does it ever make sense to cascade upstream if refframe is left? probably not
        # ? allow shifting a range of rows?
            # ! need to handl if shift leads to overlap
        
        pass
    
    @cascade
    def xshift(self, **kwargs):
        self.shift(**kwargs)
        pass

    def stitch(self):
        """Remaps variable or range of variables to another id or append rows and move to bottom (unassign)"""
        """Customized remappings based on a provided dictionary with shifting based on a range of rows"""
        # join variables arbitrarily
        # used to fix things that couldn't be automatically detected that resulted in an outer join which produced unmapped rows
        pass
    
    @cascade
    def xstitch(self, **kwargs):
        self.stitch(**kwargs)
        
        pass
    
    def build_docs(self):
        
        pass
    
    def graph(self):
		# plots the data of a given variable over time if values provided
            # ? how to handle multiple values for a variable?
        pass
    
    # 	def xlabels(self):
# 		# produces a table with the variable and all labels
# 		pass
			
# 	def lvalidate(self,summary=True,fuzzy=False):
# 		# produces a diagnostic text
# 		# produces table with labels for all previous years
# 		# if summary then only shows discrepancies, otherwise full.
# 		pass

