from __future__ import annotations

import pandas as pd
import numpy as np
import re
import difflib
import logging
from collections import namedtuple, OrderedDict
from itertools import zip_longest

from pandas import DataFrame, Series
from typing import Literal, Any, Callable, NamedTuple, overload
from IPython.display import display
from copy import copy
from csv import Sniffer

# TODO: create custom exceptions
    # on trying to manually adjust dataframes or objects rather than using provided functions

class VarMap():
            
    def __init__(self,
                 df: DataFrame,
                 vintage:str,
                 varid:str, # TODO: allow passing index
                 desc:str, 
                 *,
                 values:str|DataFrame|None=None, # if multiple values for variable use dataframe©
                 hiermethod: Literal["auto","delimited","leading","userdefined"]="auto",
                 hiercol:str|None=None,
                 pat:str|None=None,
                 orderby:str|None=None):

        # ? what if there are multiple columns that indicate hierarchy in an already parsed form?
            # expect user to pre-process?        
        # TODO: allow adding other fields that might have useful data besides description?
            # at the moment it's stored but inert
                                                    
        self._vintage = vintage        
        #self._varid = varid # ? needed ongoing? used in auto find_hierarchy
        #self._desc = desc # ? needed ongoing? used in auto find_hierarchy
            # if we rename first then it's stable                    
        self._df = df        
        self._df = self._df.rename(columns={varid:"id",desc:"desc"})  # ? order step?
        self.find_hierarchy(hiermethod,hiercol=hiercol,pat=pat,inplace=True)

        # transform dataframe to standardized format and store
            # rename value column to map_index
            # if order by is provided, that column is used to sort, otherwise assumes that default order is meaningful
                # if orderby column contains a delimited hierarchy, can't use standard sort
        # index by row and id
        #df.reset_index().set_index(varid,append=True)[desc]        


    @property
    def df(self):
        raise AttributeError(
            "Use VarMap functions to access or modify its DataFrame rather than doing so directly, or rebuild instance"
            )         

    def _repr_html_(self):
        # TODO: build general style method
        return self._df.style.set_properties(**{"text-align":"left","white-space":"pre-wrap"})._repr_html_()
            
        
    def __repr__(self):                
        return repr(self._df)

    def __lt__(self,other):
        if isinstance(other, VarMap):
            return self.vintage < other.vintage                    

    @property
    def vintage(self):
        return self._vintage
    
    @vintage.setter
    def set_vintage(self,vintage:str):
        # ! TODO: adjust DataFrame index
        self._vintage=vintage

        
    @classmethod
    def from_csv(cls,path:str,varid,desc,vintage,**kwargs) -> VarMap:
        """
        Convenience wrapper around pandas read_csv into VarMap. Kwargs are passed to read_csv.
        """
        df = pd.read_csv(path,**kwargs)
        return cls(df,vintage,varid,desc)        
    
    # @classmethod
    # def from_json(cls,path:str) -> VarMap: # TODO
        
    #     raise NotImplementedError
    
    # @staticmethod
    # def from_xlsx(path:str) -> VarMap: # TODO
        
    #     raise NotImplementedError
    
    # @staticmethod
    # def from_xml(path:str) -> VarMap: # TODO
        
    #     raise NotImplementedError
    
        
    # TODO
    # def add_values(self,df:DataFrame):
    #     # NOTE id's may be given as column names rather than rows
    
    #     pass
    
    @overload
    def find_hierarchy(self, 
                        hiermethod: Literal["auto","delimited","leading","userdefined"],                        
                        *,
                        hiercol:str|None=..., 
                        pat:str|Callable[[str],int]|None=...,                          
                        inplace:Literal[False],
                        strict:bool=True) -> VarMap: ...                        
    @overload
    def find_hierarchy(self, 
                        hiermethod: Literal["auto","delimited","leading","userdefined"],                           
                        *,
                        hiercol:str|None=..., 
                        pat:str|Callable[[str],int]|None=...,                          
                        inplace:Literal[True],
                        strict:bool=True) -> None: ...                            
    def find_hierarchy(self, 
                        hiermethod: Literal["auto","delimited","leading","userdefined"],                           
                        *,
                        hiercol:str|None=None, 
                        pat:str|Callable[[str],int]|None=None,                          
                        inplace:bool=False,
                        strict:bool=True) -> VarMap|None:
        """Used to detect nesting within the data based on a pattern in the columns and return two additional DataFrame
        columns:
            (1) level - The row's depth
            (2) v_index - A unique dot notation index that indicates the context of row's nesting
            
        Args:
            hiermethod (Literal["auto","delimited","leading","userdefined"]): Defaults to "auto".
                auto: Systematically looks for delimiters or leading characters within the id and desc fields.
                    Search order prioritizes id-delimiters, desc-leading characters, desc-delimiters. 
                    The first valid non-trivial hierarchy is returned.
                delimited: Will look for a delimiter using the provided pattern which is required. The number of splits
                    is used to help determine nesting.
                leading: Will look for leading characters based on the provided pattern which is required. Length of 
                    leading characters is used to help determine nesting.
                userdefined: Applies the user defined function provided to pattern.
            hiercol (str | None, optional): The column in which to search for a hierarchy. Must be provided if 
                hiermethod is not "auto". Defaults to None for "auto".            
            pat (str | Callable[[str],int] | None, optional): Pattern to identify nesting. A function is required if 
                hiermethod is "userdefined". Defaults to None for "auto" hiermethod.
            inplace (bool, optional): If inplace then the VarMap's DataFrame is modified. 
                If not inplace then a new VarMap is returned with a DataFrame including the new fields. 
                Defaults to False.
            strict (bool, optional): If strict then only hierarchies that meet the following are accepted:
                1. Begins at level 1; 2. All rows exactly match the pattern; 3. Levels cannot increase more than 1 per
                row. Defaults to True.

        Returns:
            VarMap|None
        """            
        # TODO: json inference method from nesting        
        
        if hiermethod!="auto":
            if hiercol is None or pat is None:
                raise ValueError("If not using 'auto' hiermethod then hiercol and pat must be provided")          
            if hiermethod=="userdefined" and not callable(pat):
                raise TypeError("A Callable must be provided hwen using 'userdefined' hiermethod")
                
        df = self._df.copy()
        
        class InferStep(NamedTuple):
            method: str
            hierser: Series                        
            
                    
        if hiermethod=="auto":            
            steps = [
                InferStep("delimited",df["id"]),
                InferStep("leading",df["desc"]),
                InferStep("delimited",df["desc"]),                
            ]
        else:                        
                steps = [InferStep(hiermethod,df[hiercol])]

        for step in steps:                        
            hierser = step.hierser.astype(str).fillna("")
            if hiermethod=="auto": # find pat
                if step.method=="delimited":                
                    s_delim = hierser.apply(lambda x: Sniffer().sniff(x.replace(" ","")).delimiter)
                    pat = s_delim.mode()[0]
                    # check for repeated delimiters (e.g. !!! instead of just !)
                    pat = hierser.apply(lambda x: re.findall(f"{pat}+",x)).explode().mode()[0]                                                
                elif step.method=="leading":                                                
                    pat = " └├│"
        
            if step.method=="delimited":
                algo = lambda rowstr: len(rowstr.split(pat))
            elif step.method=="leading":            
                algo = lambda rowstr: len(rowstr) - len(rowstr.lstrip(pat))
            elif step.method=="userdefined":
                algo = lambda rowstr: pat(rowstr)
            
            lvl_indic = hierser.map(algo)            
            # simplify lvl indicators
            indic_diffs = lvl_indic - lvl_indic.shift(1).fillna(0,limit=1) # first row shift is null to be treated as 0
            reducer = indic_diffs[indic_diffs>0].abs().mode()[0] # most common diff should be the one-step change
            lvls = lvl_indic.floordiv(reducer).astype(int)+1 # floordiv to handle anomalies that don't divide evenly                                      
            lvl_diffs = (lvls - lvls.shift(1).fillna(0,limit=1))
            
            # validation            
            # look for levels not starting at level 1, that have indicator anomalies anomalies, or level jumps            
            if (lvls.iloc[0]!=1) or (lvls.ne(lvl_indic.div(reducer).fillna(0,limit=1)+1).any()) or ((lvl_diffs>1).any()):
                if hiermethod!="auto": # only non-auto cases provide feedback
                    if strict:
                        # TODO: provide specific problem and row where error occurs
                        raise Exception("Invalid hierarchy")
                    else:                        
                        logging.warn("WARNING: Could not find a valid hierarchy - returning flat")
                
                lvls = Series([1]*lvls.size) # replace with flattened
                           
            if lvls.unique().size > 1: # exit loop when valid non-trivial hierarchy is found                
                df["level"] = lvls
                break                    
        print(lvls.unique())
        # use level data to create vectorized dot notation index
        # represent each level as a list for additive purposes
        lvl_adds = lvls.apply(lambda x: [0]*(x-1)+[1]).apply(Series)
        # cumulatively sum layers but reset sub-levels when a higher level increases
        vectors = lvl_adds.apply(lambda x: x.groupby(x.isna().cumsum()).cumsum(),axis=0)
        df["v_index"] = vectors.fillna(0).apply(lambda x: ".".join([str(int(v)) for v in x if v>0]),axis=1)                    
        
        # TODO: implement a well-formatted description based on level
            # delimited: content = x["desc"].apply(lambda x: x.split(pat)[x["level"]])
            # leading: content = x["desc"].apply(lambda x: x.lstrip(pat))
            # userdefined: ? the userdefined returns a level so not clear how to structure the content
                # just treat it like leading?
        #df["desc2"] = df.apply(lambda x: " "*((x["level"]*2)-1)+x["content"]),axis=1)
            
        
        if inplace:            
            self._df = df
            return None
        else:            
            vm = copy(self)
            vm._df = df
            return vm       
    

class XVarMap():
    
    def __init__(self, varmaps:list[VarMap],refframe):
        
        self._refframe = refframe        
        self._varmaps = varmaps # store as dict with the vintage as key?
        
        
        
        # join varmaps on id and index by refframe's row
        
        
        # full data
            # grouped columns by id, description, depth, summary and then granular by vintage
            # can run versions of the full data that only show certain groups
            
        # maybe streamlit can help?
            
        # each cell stores the id, description, and depth
        
        # TODO: vintages list property
        
        pass
    
    @property
    def varmaps(self):                
        raise AttributeError("""
            Use XVarMap functions to access or modify its VarMap objects rather than doing so directly, 
            or rebuild instance"
            """)        
    
    # @classmethod
    # def from_csv(cls,path:str|list[str],**kwargs) -> None:
    #     # TODO: convenience wrapper around VarMap.from_csv for multiple
    #     # list can be a regex string of matching filenames
    #     # ? what happens on an error?
    #         # complicated to pass kwargs in the event that not everything in consistent in infer hierarchy step
    #         # need to also pass a mapping of vintages or a list that is as long as the number of files        
        
    #     pass
        
    def save(self,path:str):
        
        pass

    # TODO: prevent multiple VarMap's with the same vintage from being added
        # attach to method that adds VarMaps 
    # TODO: when changing vintage of VarMap, need to check that it doesn't lead to multiple VarMaps with the same vintage being in an XVarMap
        # create a set_vintages property that takes a mapping of old vintage names to new vintages with the check for no duplicates before updating
            # ! but the list is accessible
                # make varmaps a property to return the individual varmaps
                # then after accessing them run the validation code?
        
    
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
            raise ValueError("""
                             'vintage' cannot be the frame of reference. This transform is done relative to the frame 
                             of reference
                             """)
    
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
    
    def cascade(self, mode:Literal["relative","subset","upstream","downstream"]="relative", subset:list[Any]|None=None):
        
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

