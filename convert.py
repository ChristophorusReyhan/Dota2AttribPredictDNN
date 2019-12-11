from html.parser import HTMLParser
import sys
import re

class HTMLTableParser(HTMLParser):
    def __init__(self, row_delim="\n", cell_delim=","):
        HTMLParser.__init__(self)
        self.despace_re = re.compile("\s+")
        self.data_interrupt = False
        self.first_row = True
        self.first_cell = True
        self.in_cell = False
        self.row_delim = row_delim
        self.cell_delim = cell_delim
        self.quote_buffer = False
        self.buffer = None

    def handle_starttag(self, tag, attrs):
        self.data_interrupt = True
        if tag == "table":
            self.first_row = True
            self.first_cell = True
        elif tag == "tr":
            if not self.first_row:
                sys.stdout.write(self.row_delim)
            self.first_row = False
            self.first_cell = True
            self.data_interrupt = False
        elif tag == "td" or tag == "th":
            if not self.first_cell:
                sys.stdout.write(self.cell_delim)
            self.first_cell = False
            self.data_interrupt = False
            self.in_cell = True
        elif tag == "br":
            self.quote_buffer = True
            self.buffer += self.row_delim

    def handle_endtag(self, tag):
        self.data_interrupt = True
        if tag == "td" or tag == "th":
            self.in_cell = False
        if self.buffer != None:
            # Quote if needed...
            if self.quote_buffer or self.cell_delim in self.buffer or "\"" in self.buffer:
                # Need to quote! First, replace all double-quotes with quad-quotes
                self.buffer = self.buffer.replace("\"", "\"\"")
                self.buffer = "\"{0}\"".format(self.buffer)
            sys.stdout.write(self.buffer)
            self.quote_buffer = False
            self.buffer = None

    def handle_data(self, data):
        if self.in_cell:
            #if self.data_interrupt:
            #   sys.stdout.write(" ")
            if self.buffer == None:
                self.buffer = ""
            self.buffer += self.despace_re.sub(" ", data).strip()
            self.data_interrupt = False

parser = HTMLTableParser() 
tbl = '''
<table class="evenrowsgray wikitable sortable jquery-tablesorter" style="width:100%; text-align:center; font-size:88%;">
<thead><tr>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending">HERO
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Primary attribute" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">A</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Base strength" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">STR</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Strength growth per level" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">STR+</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Total strength at level 25 (no bonus)" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">STR<br>25</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">AGI</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Agility growth per level" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">AGI+</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Total agility at level 25 (no bonus)" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">AGI<br>25</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Base intelligence" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">INT</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Intelligence growth per level" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">INT+</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Total intelligence at level 25 (no bonus)" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">INT<br>25</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Total starting attributes" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">T</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Total attribute growth per level" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">T+</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Total attributes at level 25 (no bonus)" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">T25</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Base movement speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">MS</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Starting armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">AR</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Starting attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">DMG(MIN)</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Starting attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">DMG(MAX)</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Attack range" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">RG</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Attack speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">AS</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Base attack time" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">BAT</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Attack point" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">ATK<br>PT</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Attack backswing" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">ATK<br>BS</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Vision range during daytime" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">VS-D</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Vision range during nighttime" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">VS-N</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Turn rate" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">TR</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Collision size" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">COL</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Base health regeneration" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">HP/S</span>
</th>
<th class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><span class="tooltip" title="Legs" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">L</span>
</th></tr></thead><tbody>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Abaddon" title="Abaddon"><img alt="Abaddon minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/cc/Abaddon_minimap_icon.png/20px-Abaddon_minimap_icon.png?version=226cae6432aecc30e6566b7423791fec" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/cc/Abaddon_minimap_icon.png/30px-Abaddon_minimap_icon.png?version=226cae6432aecc30e6566b7423791fec 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/c/cc/Abaddon_minimap_icon.png?version=226cae6432aecc30e6566b7423791fec 2x"></a> <a href="/Abaddon" title="Abaddon">Abaddon</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>3</td>
<td>95</td>
<td>23</td>
<td>1.5</td>
<td>59</td>
<td>18</td>
<td>2</td>
<td>66</td>
<td>64</td>
<td>6.5</td>
<td>220</td>
<td><span class="tooltip" title="328.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">325</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.68</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="40 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>120</td>
<td>1.7</td>
<td>0.56</td>
<td>0.41</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Alchemist" title="Alchemist"><img alt="Alchemist minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1e/Alchemist_minimap_icon.png/20px-Alchemist_minimap_icon.png?version=3e83871e21f926d69372e7c2066450e4" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1e/Alchemist_minimap_icon.png/30px-Alchemist_minimap_icon.png?version=3e83871e21f926d69372e7c2066450e4 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/1e/Alchemist_minimap_icon.png?version=3e83871e21f926d69372e7c2066450e4 2x"></a> <a href="/Alchemist" title="Alchemist">Alchemist</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>25</td>
<td>2.4</td>
<td>82.6</td>
<td>22</td>
<td>1.2</td>
<td>50.8</td>
<td>25</td>
<td>1.8</td>
<td>68.2</td>
<td>72</td>
<td>5.4</td>
<td>201.6</td>
<td><span class="tooltip" title="308.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.52</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">58</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0.65</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Ancient_Apparition" title="Ancient Apparition"><img alt="Ancient Apparition minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/25/Ancient_Apparition_minimap_icon.png/20px-Ancient_Apparition_minimap_icon.png?version=bc95ee5dea4d688c731bab8ecdeeeeb2" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/25/Ancient_Apparition_minimap_icon.png/30px-Ancient_Apparition_minimap_icon.png?version=bc95ee5dea4d688c731bab8ecdeeeeb2 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/25/Ancient_Apparition_minimap_icon.png?version=bc95ee5dea4d688c731bab8ecdeeeeb2 2x"></a> <a href="/Ancient_Apparition" title="Ancient Apparition">Ancient Apparition</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>20</td>
<td>1.9</td>
<td>65.6</td>
<td>20</td>
<td>2.2</td>
<td>72.8</td>
<td>23</td>
<td>3.4</td>
<td>104.6</td>
<td>63</td>
<td>7.5</td>
<td>243</td>
<td><span class="tooltip" title="287.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.2</span></td>
<td><span class="tooltip" title="21 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="1250 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">675</span></td>
<td>100</td>
<td>1.7</td>
<td>0.45</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Anti-Mage" title="Anti-Mage"><img alt="Anti-Mage minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/30/Anti-Mage_minimap_icon.png/20px-Anti-Mage_minimap_icon.png?version=a4f23c0f7e26ee8255418a8ab3ce8395" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/30/Anti-Mage_minimap_icon.png/30px-Anti-Mage_minimap_icon.png?version=a4f23c0f7e26ee8255418a8ab3ce8395 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/3/30/Anti-Mage_minimap_icon.png?version=a4f23c0f7e26ee8255418a8ab3ce8395 2x"></a> <a href="/Anti-Mage" title="Anti-Mage">Anti-Mage</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>23</td>
<td>1.3</td>
<td>54.2</td>
<td>24</td>
<td>3</td>
<td>96</td>
<td>12</td>
<td>1.8</td>
<td>55.2</td>
<td>59</td>
<td>6.1</td>
<td>205.4</td>
<td><span class="tooltip" title="313.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.84</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.4</td>
<td>0.3</td>
<td>0.6</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Arc_Warden" title="Arc Warden"><img alt="Arc Warden minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/e9/Arc_Warden_minimap_icon.png/20px-Arc_Warden_minimap_icon.png?version=5b4171b298dace6395e393c3236f50e8" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/e9/Arc_Warden_minimap_icon.png/30px-Arc_Warden_minimap_icon.png?version=5b4171b298dace6395e393c3236f50e8 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/e/e9/Arc_Warden_minimap_icon.png?version=5b4171b298dace6395e393c3236f50e8 2x"></a> <a href="/Arc_Warden" title="Arc Warden">Arc Warden</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>25</td>
<td>3</td>
<td>97</td>
<td>15</td>
<td>2.4</td>
<td>72.6</td>
<td>24</td>
<td>2.6</td>
<td>86.4</td>
<td>64</td>
<td>8</td>
<td>256</td>
<td><span class="tooltip" title="282.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0.4</span></td>
<td><span class="tooltip" title="31 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">625</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Axe" title="Axe"><img alt="Axe minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/f7/Axe_minimap_icon.png/20px-Axe_minimap_icon.png?version=591ca9a98e6ff4edf5c8c2b1d9997ecf" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/f/f7/Axe_minimap_icon.png?version=591ca9a98e6ff4edf5c8c2b1d9997ecf 1.5x"></a> <a href="/Axe" title="Axe">Axe</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>25</td>
<td>3.4</td>
<td>106.6</td>
<td>20</td>
<td>2.2</td>
<td>72.8</td>
<td>18</td>
<td>1.6</td>
<td>56.4</td>
<td>63</td>
<td>7.2</td>
<td>235.8</td>
<td><span class="tooltip" title="308.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.2</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">52</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>2.75</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Bane" title="Bane"><img alt="Bane minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0d/Bane_minimap_icon.png/20px-Bane_minimap_icon.png?version=72a4573f3516f6762aa6ef7e77998346" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0d/Bane_minimap_icon.png/30px-Bane_minimap_icon.png?version=72a4573f3516f6762aa6ef7e77998346 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/0d/Bane_minimap_icon.png?version=72a4573f3516f6762aa6ef7e77998346 2x"></a> <a href="/Bane" title="Bane">Bane</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>23</td>
<td>2.6</td>
<td>85.4</td>
<td>23</td>
<td>2.6</td>
<td>85.4</td>
<td>23</td>
<td>2.6</td>
<td>85.4</td>
<td>69</td>
<td>7.8</td>
<td>256.2</td>
<td><span class="tooltip" title="308.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.68</span></td>
<td><span class="tooltip" title="35 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">58</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">64</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">400</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>4</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Batrider" title="Batrider"><img alt="Batrider minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0d/Batrider_minimap_icon.png/20px-Batrider_minimap_icon.png?version=fbcedaab12c49ad0b98612881f332033" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0d/Batrider_minimap_icon.png/30px-Batrider_minimap_icon.png?version=fbcedaab12c49ad0b98612881f332033 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/0d/Batrider_minimap_icon.png?version=fbcedaab12c49ad0b98612881f332033 2x"></a> <a href="/Batrider" title="Batrider">Batrider</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>28</td>
<td>2.9</td>
<td>97.6</td>
<td>15</td>
<td>1.5</td>
<td>51</td>
<td>22</td>
<td>2.9</td>
<td>91.6</td>
<td>65</td>
<td>7.3</td>
<td>240.2</td>
<td><span class="tooltip" title="292.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.4</span></td>
<td><span class="tooltip" title="16 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">38</span></td>
<td><span class="tooltip" title="20 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">375</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.54</td>
<td>1200</td>
<td>800</td>
<td>1</td>
<td>24</td>
<td>1.75</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Beastmaster" title="Beastmaster"><img alt="Beastmaster minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5c/Beastmaster_minimap_icon.png/20px-Beastmaster_minimap_icon.png?version=c70c00a25b8edd2d8f1ffd7251db0e2c" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5c/Beastmaster_minimap_icon.png/30px-Beastmaster_minimap_icon.png?version=c70c00a25b8edd2d8f1ffd7251db0e2c 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/5c/Beastmaster_minimap_icon.png?version=c70c00a25b8edd2d8f1ffd7251db0e2c 2x"></a> <a href="/Beastmaster" title="Beastmaster">Beastmaster</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>2.9</td>
<td>92.6</td>
<td>18</td>
<td>1.6</td>
<td>56.4</td>
<td>16</td>
<td>1.9</td>
<td>61.6</td>
<td>57</td>
<td>6.4</td>
<td>210.6</td>
<td><span class="tooltip" title="307.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.88</span></td>
<td><span class="tooltip" title="41 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">64</span></td>
<td><span class="tooltip" title="45 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">68</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Bloodseeker" title="Bloodseeker"><img alt="Bloodseeker minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/57/Bloodseeker_minimap_icon.png/20px-Bloodseeker_minimap_icon.png?version=407b7418caa38b36d53627aa97aa54e8" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/57/Bloodseeker_minimap_icon.png/30px-Bloodseeker_minimap_icon.png?version=407b7418caa38b36d53627aa97aa54e8 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/57/Bloodseeker_minimap_icon.png?version=407b7418caa38b36d53627aa97aa54e8 2x"></a> <a href="/Bloodseeker" title="Bloodseeker">Bloodseeker</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>24</td>
<td>2.7</td>
<td>88.8</td>
<td>22</td>
<td>3.4</td>
<td>103.6</td>
<td>18</td>
<td>1.7</td>
<td>58.8</td>
<td>64</td>
<td>7.8</td>
<td>251.2</td>
<td><span class="tooltip" title="303.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.52</span></td>
<td><span class="tooltip" title="33 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">55</span></td>
<td><span class="tooltip" title="39 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.43</td>
<td>0.74</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Bounty_Hunter" title="Bounty Hunter"><img alt="Bounty Hunter minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9c/Bounty_Hunter_minimap_icon.png/20px-Bounty_Hunter_minimap_icon.png?version=4007b529de4e09235e5a351d73dba76d" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9c/Bounty_Hunter_minimap_icon.png/30px-Bounty_Hunter_minimap_icon.png?version=4007b529de4e09235e5a351d73dba76d 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/9c/Bounty_Hunter_minimap_icon.png?version=4007b529de4e09235e5a351d73dba76d 2x"></a> <a href="/Bounty_Hunter" title="Bounty Hunter">Bounty Hunter</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>20</td>
<td>2.5</td>
<td>80</td>
<td>21</td>
<td>2.6</td>
<td>83.4</td>
<td>19</td>
<td>2</td>
<td>67</td>
<td>60</td>
<td>7.1</td>
<td>230.4</td>
<td><span class="tooltip" title="318.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">315</span></td>
<td><span class="tooltip" title="3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">6.36</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">48</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">62</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.59</td>
<td>0.59</td>
<td>1800</td>
<td>1000</td>
<td>0.6</td>
<td>24</td>
<td>1.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Brewmaster" title="Brewmaster"><img alt="Brewmaster minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9b/Brewmaster_minimap_icon.png/20px-Brewmaster_minimap_icon.png?version=eaa578c82793896a162f3188754b9236" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9b/Brewmaster_minimap_icon.png/30px-Brewmaster_minimap_icon.png?version=eaa578c82793896a162f3188754b9236 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/9b/Brewmaster_minimap_icon.png?version=eaa578c82793896a162f3188754b9236 2x"></a> <a href="/Brewmaster" title="Brewmaster">Brewmaster</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>3.7</td>
<td>111.8</td>
<td>22</td>
<td>2</td>
<td>70</td>
<td>15</td>
<td>1.6</td>
<td>53.4</td>
<td>60</td>
<td>7.3</td>
<td>235.2</td>
<td><span class="tooltip" title="308.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.52</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">52</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0.65</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Bristleback" title="Bristleback"><img alt="Bristleback minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a8/Bristleback_minimap_icon.png/20px-Bristleback_minimap_icon.png?version=87c49172aa28f0a30aa4e5fdbd1c3d10" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a8/Bristleback_minimap_icon.png/30px-Bristleback_minimap_icon.png?version=87c49172aa28f0a30aa4e5fdbd1c3d10 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/a/a8/Bristleback_minimap_icon.png?version=87c49172aa28f0a30aa4e5fdbd1c3d10 2x"></a> <a href="/Bristleback" title="Bristleback">Bristleback</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>2.9</td>
<td>91.6</td>
<td>17</td>
<td>1.8</td>
<td>60.2</td>
<td>14</td>
<td>2.8</td>
<td>81.2</td>
<td>53</td>
<td>7.5</td>
<td>233</td>
<td><span class="tooltip" title="292.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.72</span></td>
<td><span class="tooltip" title="25 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.8</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>1</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Broodmother" title="Broodmother"><img alt="Broodmother minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/19/Broodmother_minimap_icon.png/20px-Broodmother_minimap_icon.png?version=9bf171f4ede74b2c3c0393a318dee73c" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/19/Broodmother_minimap_icon.png/30px-Broodmother_minimap_icon.png?version=9bf171f4ede74b2c3c0393a318dee73c 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/19/Broodmother_minimap_icon.png?version=9bf171f4ede74b2c3c0393a318dee73c 2x"></a> <a href="/Broodmother" title="Broodmother">Broodmother</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>18</td>
<td>2.8</td>
<td>85.2</td>
<td>15</td>
<td>2.8</td>
<td>82.2</td>
<td>18</td>
<td>2</td>
<td>66</td>
<td>51</td>
<td>7.6</td>
<td>233.4</td>
<td><span class="tooltip" title="277.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">275</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.4</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>125</td>
<td>1.7</td>
<td>0.4</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>8</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Centaur_Warrunner" title="Centaur Warrunner"><img alt="Centaur Warrunner minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5e/Centaur_Warrunner_minimap_icon.png/20px-Centaur_Warrunner_minimap_icon.png?version=b2c4087f0448c68e38beee0dbac52d32" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5e/Centaur_Warrunner_minimap_icon.png/30px-Centaur_Warrunner_minimap_icon.png?version=b2c4087f0448c68e38beee0dbac52d32 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/5e/Centaur_Warrunner_minimap_icon.png?version=b2c4087f0448c68e38beee0dbac52d32 2x"></a> <a href="/Centaur_Warrunner" title="Centaur Warrunner">Centaur Warrunner</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>27</td>
<td>4.6</td>
<td>137.4</td>
<td>15</td>
<td>1</td>
<td>39</td>
<td>15</td>
<td>1.6</td>
<td>53.4</td>
<td>57</td>
<td>7.2</td>
<td>229.8</td>
<td><span class="tooltip" title="302.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.4</span></td>
<td><span class="tooltip" title="36 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="38 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">65</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>4</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Chaos_Knight" title="Chaos Knight"><img alt="Chaos Knight minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9f/Chaos_Knight_minimap_icon.png/20px-Chaos_Knight_minimap_icon.png?version=090b765a980bc40bcc17eae96b1e3ccf" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9f/Chaos_Knight_minimap_icon.png/30px-Chaos_Knight_minimap_icon.png?version=090b765a980bc40bcc17eae96b1e3ccf 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/9f/Chaos_Knight_minimap_icon.png?version=090b765a980bc40bcc17eae96b1e3ccf 2x"></a> <a href="/Chaos_Knight" title="Chaos Knight">Chaos Knight</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>3.4</td>
<td>103.6</td>
<td>14</td>
<td>1.4</td>
<td>47.6</td>
<td>18</td>
<td>1.2</td>
<td>46.8</td>
<td>54</td>
<td>6</td>
<td>198</td>
<td><span class="tooltip" title="322.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">320</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.24</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="59 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">81</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Chen" title="Chen"><img alt="Chen minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/fc/Chen_minimap_icon.png/20px-Chen_minimap_icon.png?version=4cbe1d98ff1a47359694a2f246b79196" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/fc/Chen_minimap_icon.png/30px-Chen_minimap_icon.png?version=4cbe1d98ff1a47359694a2f246b79196 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/f/fc/Chen_minimap_icon.png?version=4cbe1d98ff1a47359694a2f246b79196 2x"></a> <a href="/Chen" title="Chen">Chen</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>25</td>
<td>2</td>
<td>73</td>
<td>15</td>
<td>2.1</td>
<td>65.4</td>
<td>19</td>
<td>3.2</td>
<td>95.8</td>
<td>59</td>
<td>7.3</td>
<td>234.2</td>
<td><span class="tooltip" title="302.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.4</span></td>
<td><span class="tooltip" title="25 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="1100 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">650</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Clinkz" title="Clinkz"><img alt="Clinkz minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/16/Clinkz_minimap_icon.png/20px-Clinkz_minimap_icon.png?version=5ed503231e3f1f53d6f832d8cac48244" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/16/Clinkz_minimap_icon.png/30px-Clinkz_minimap_icon.png?version=5ed503231e3f1f53d6f832d8cac48244 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/16/Clinkz_minimap_icon.png?version=5ed503231e3f1f53d6f832d8cac48244 2x"></a> <a href="/Clinkz" title="Clinkz">Clinkz</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>14</td>
<td>2.2</td>
<td>66.8</td>
<td>22</td>
<td>2.7</td>
<td>86.8</td>
<td>18</td>
<td>1.7</td>
<td>58.8</td>
<td>54</td>
<td>6.6</td>
<td>212.4</td>
<td><span class="tooltip" title="293.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.52</span></td>
<td><span class="tooltip" title="15 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">37</span></td>
<td><span class="tooltip" title="21 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">43</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">625</span></td>
<td>100</td>
<td>1.7</td>
<td>0.7</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Clockwerk" title="Clockwerk"><img alt="Clockwerk minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/12/Clockwerk_minimap_icon.png/20px-Clockwerk_minimap_icon.png?version=33b9b7bd1d9fa764927d0c79c017d96e" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/12/Clockwerk_minimap_icon.png/30px-Clockwerk_minimap_icon.png?version=33b9b7bd1d9fa764927d0c79c017d96e 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/12/Clockwerk_minimap_icon.png?version=33b9b7bd1d9fa764927d0c79c017d96e 2x"></a> <a href="/Clockwerk" title="Clockwerk">Clockwerk</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>26</td>
<td>3.7</td>
<td>114.8</td>
<td>13</td>
<td>2.3</td>
<td>68.2</td>
<td>18</td>
<td>1.5</td>
<td>54</td>
<td>57</td>
<td>7.5</td>
<td>237</td>
<td><span class="tooltip" title="317 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">315</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.08</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="26 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">52</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.33</td>
<td>0.64</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Crystal_Maiden" title="Crystal Maiden"><img alt="Crystal Maiden minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b4/Crystal_Maiden_minimap_icon.png/20px-Crystal_Maiden_minimap_icon.png?version=affe6b1733ba7643374b3496356b2321" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b4/Crystal_Maiden_minimap_icon.png/30px-Crystal_Maiden_minimap_icon.png?version=affe6b1733ba7643374b3496356b2321 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/b4/Crystal_Maiden_minimap_icon.png?version=affe6b1733ba7643374b3496356b2321 2x"></a> <a href="/Crystal_Maiden" title="Crystal Maiden">Crystal Maiden</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.2</td>
<td>70.8</td>
<td>16</td>
<td>1.6</td>
<td>54.4</td>
<td>14</td>
<td>3.3</td>
<td>93.2</td>
<td>48</td>
<td>7.1</td>
<td>218.4</td>
<td><span class="tooltip" title="277.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">275</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.56</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>115</td>
<td>1.7</td>
<td>0.45</td>
<td>0</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Dark_Seer" title="Dark Seer"><img alt="Dark Seer minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5a/Dark_Seer_minimap_icon.png/20px-Dark_Seer_minimap_icon.png?version=b30911f75f54957350f07958f040ba0c" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5a/Dark_Seer_minimap_icon.png/30px-Dark_Seer_minimap_icon.png?version=b30911f75f54957350f07958f040ba0c 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/5a/Dark_Seer_minimap_icon.png?version=b30911f75f54957350f07958f040ba0c 2x"></a> <a href="/Dark_Seer" title="Dark Seer">Dark Seer</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>20</td>
<td>3.1</td>
<td>94.4</td>
<td>12</td>
<td>1.8</td>
<td>55.2</td>
<td>21</td>
<td>3.1</td>
<td>95.4</td>
<td>53</td>
<td>8</td>
<td>245</td>
<td><span class="tooltip" title="296.8 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.92</span></td>
<td><span class="tooltip" title="33 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="39 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.59</td>
<td>0.58</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Dark_Willow" title="Dark Willow"><img alt="Dark Willow minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/47/Dark_Willow_minimap_icon.png/20px-Dark_Willow_minimap_icon.png?version=8744b8757ed309d3a0743ca53fbe7d82" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/47/Dark_Willow_minimap_icon.png/30px-Dark_Willow_minimap_icon.png?version=8744b8757ed309d3a0743ca53fbe7d82 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/4/47/Dark_Willow_minimap_icon.png?version=8744b8757ed309d3a0743ca53fbe7d82 2x"></a> <a href="/Dark_Willow" title="Dark Willow">Dark Willow</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>20</td>
<td>2</td>
<td>68</td>
<td>18</td>
<td>1.6</td>
<td>56.4</td>
<td>18</td>
<td>3.5</td>
<td>102</td>
<td>56</td>
<td>7.1</td>
<td>226.4</td>
<td><span class="tooltip" title="292.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.88</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">475</span></td>
<td>115</td>
<td>1.5</td>
<td>0.3</td>
<td>0</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Dazzle" title="Dazzle"><img alt="Dazzle minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/3b/Dazzle_minimap_icon.png/20px-Dazzle_minimap_icon.png?version=9555c4571e368f52cfd48e12e918481f" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/3b/Dazzle_minimap_icon.png/30px-Dazzle_minimap_icon.png?version=9555c4571e368f52cfd48e12e918481f 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/3/3b/Dazzle_minimap_icon.png?version=9555c4571e368f52cfd48e12e918481f 2x"></a> <a href="/Dazzle" title="Dazzle">Dazzle</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.5</td>
<td>78</td>
<td>21</td>
<td>1.7</td>
<td>61.8</td>
<td>25</td>
<td>3.7</td>
<td>113.8</td>
<td>64</td>
<td>7.9</td>
<td>253.6</td>
<td><span class="tooltip" title="308.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.36</span></td>
<td><span class="tooltip" title="22 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="28 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Death_Prophet" title="Death Prophet"><img alt="Death Prophet minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5d/Death_Prophet_minimap_icon.png/20px-Death_Prophet_minimap_icon.png?version=18f46e25d6d742db75a334d6d1386033" width="20" height="21" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/5/5d/Death_Prophet_minimap_icon.png?version=18f46e25d6d742db75a334d6d1386033 1.5x"></a> <a href="/Death_Prophet" title="Death Prophet">Death Prophet</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>19</td>
<td>3.1</td>
<td>93.4</td>
<td>14</td>
<td>1.4</td>
<td>47.6</td>
<td>24</td>
<td>3.5</td>
<td>108</td>
<td>57</td>
<td>8</td>
<td>249</td>
<td><span class="tooltip" title="312.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.24</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">65</span></td>
<td><span class="tooltip" title="1000 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.56</td>
<td>0.51</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Disruptor" title="Disruptor"><img alt="Disruptor minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b9/Disruptor_minimap_icon.png/20px-Disruptor_minimap_icon.png?version=46538dc48264f0a6c335bebbd1ec50b8" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b9/Disruptor_minimap_icon.png/30px-Disruptor_minimap_icon.png?version=46538dc48264f0a6c335bebbd1ec50b8 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/b9/Disruptor_minimap_icon.png?version=46538dc48264f0a6c335bebbd1ec50b8 2x"></a> <a href="/Disruptor" title="Disruptor">Disruptor</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2.4</td>
<td>78.6</td>
<td>15</td>
<td>1.4</td>
<td>48.6</td>
<td>20</td>
<td>2.9</td>
<td>89.6</td>
<td>56</td>
<td>6.7</td>
<td>216.8</td>
<td><span class="tooltip" title="297.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.4</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">625</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Doom" title="Doom"><img alt="Doom minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/96/Doom_minimap_icon.png/20px-Doom_minimap_icon.png?version=da5a14b15303765da5110051a26c6c9e" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/96/Doom_minimap_icon.png/30px-Doom_minimap_icon.png?version=da5a14b15303765da5110051a26c6c9e 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/96/Doom_minimap_icon.png?version=da5a14b15303765da5110051a26c6c9e 2x"></a> <a href="/Doom" title="Doom">Doom</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>26</td>
<td>4</td>
<td>122</td>
<td>11</td>
<td>0.9</td>
<td>32.6</td>
<td>15</td>
<td>2.1</td>
<td>65.4</td>
<td>52</td>
<td>7</td>
<td>220</td>
<td><span class="tooltip" title="286.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0.76</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="46 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">72</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">175</span></td>
<td>100</td>
<td>2</td>
<td>0.5</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Dragon_Knight" title="Dragon Knight"><img alt="Dragon Knight minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/21/Dragon_Knight_minimap_icon.png/20px-Dragon_Knight_minimap_icon.png?version=7537ccf45dae1a118e671bcd8fefa06f" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/21/Dragon_Knight_minimap_icon.png/30px-Dragon_Knight_minimap_icon.png?version=7537ccf45dae1a118e671bcd8fefa06f 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/21/Dragon_Knight_minimap_icon.png?version=7537ccf45dae1a118e671bcd8fefa06f 2x"></a> <a href="/Dragon_Knight" title="Dragon Knight">Dragon Knight</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>19</td>
<td>3.6</td>
<td>105.4</td>
<td>19</td>
<td>2</td>
<td>67</td>
<td>18</td>
<td>1.7</td>
<td>58.8</td>
<td>56</td>
<td>7.3</td>
<td>231.2</td>
<td><span class="tooltip" title="297.8 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.04</span></td>
<td><span class="tooltip" title="31 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="37 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Drow_Ranger" title="Drow Ranger"><img alt="Drow Ranger minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/63/Drow_Ranger_minimap_icon.png/20px-Drow_Ranger_minimap_icon.png?version=b777d21d063ecd284015b1130987e03c" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/63/Drow_Ranger_minimap_icon.png/30px-Drow_Ranger_minimap_icon.png?version=b777d21d063ecd284015b1130987e03c 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/63/Drow_Ranger_minimap_icon.png?version=b777d21d063ecd284015b1130987e03c 2x"></a> <a href="/Drow_Ranger" title="Drow Ranger">Drow Ranger</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>18</td>
<td>1.9</td>
<td>63.6</td>
<td>20</td>
<td>2.9</td>
<td>89.6</td>
<td>15</td>
<td>1.4</td>
<td>48.6</td>
<td>53</td>
<td>6.2</td>
<td>201.8</td>
<td><span class="tooltip" title="287.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="-3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0.2</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="40 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="1250 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">625</span></td>
<td>100</td>
<td>1.7</td>
<td>0.65</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Earth_Spirit" title="Earth Spirit"><img alt="Earth Spirit minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1f/Earth_Spirit_minimap_icon.png/20px-Earth_Spirit_minimap_icon.png?version=23d72fcffb9b037a2b75cb83cb207e2e" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1f/Earth_Spirit_minimap_icon.png/30px-Earth_Spirit_minimap_icon.png?version=23d72fcffb9b037a2b75cb83cb207e2e 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/1f/Earth_Spirit_minimap_icon.png?version=23d72fcffb9b037a2b75cb83cb207e2e 2x"></a> <a href="/Earth_Spirit" title="Earth Spirit">Earth Spirit</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>21</td>
<td>4.4</td>
<td>126.6</td>
<td>17</td>
<td>1.5</td>
<td>53</td>
<td>20</td>
<td>2.1</td>
<td>70.4</td>
<td>58</td>
<td>8</td>
<td>250</td>
<td><span class="tooltip" title="292.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.72</span></td>
<td><span class="tooltip" title="25 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0.65</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Earthshaker" title="Earthshaker"><img alt="Earthshaker minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/42/Earthshaker_minimap_icon.png/20px-Earthshaker_minimap_icon.png?version=8992c1e487855ae7539c387ff51ab4c6" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/42/Earthshaker_minimap_icon.png/30px-Earthshaker_minimap_icon.png?version=8992c1e487855ae7539c387ff51ab4c6 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/4/42/Earthshaker_minimap_icon.png?version=8992c1e487855ae7539c387ff51ab4c6 2x"></a> <a href="/Earthshaker" title="Earthshaker">Earthshaker</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>3.7</td>
<td>110.8</td>
<td>12</td>
<td>1.4</td>
<td>45.6</td>
<td>16</td>
<td>1.8</td>
<td>59.2</td>
<td>50</td>
<td>6.9</td>
<td>215.6</td>
<td><span class="tooltip" title="311.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.92</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="37 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.467</td>
<td>0.863</td>
<td>1800</td>
<td>800</td>
<td>0.9</td>
<td>24</td>
<td>1</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Elder_Titan" title="Elder Titan"><img alt="Elder Titan minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9f/Elder_Titan_minimap_icon.png/20px-Elder_Titan_minimap_icon.png?version=0bcf1e8c58175a0880c659dd72736e50" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9f/Elder_Titan_minimap_icon.png/30px-Elder_Titan_minimap_icon.png?version=0bcf1e8c58175a0880c659dd72736e50 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/9f/Elder_Titan_minimap_icon.png?version=0bcf1e8c58175a0880c659dd72736e50 2x"></a> <a href="/Elder_Titan" title="Elder Titan">Elder Titan</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>24</td>
<td>3</td>
<td>96</td>
<td>14</td>
<td>1.8</td>
<td>57.2</td>
<td>23</td>
<td>1.6</td>
<td>61.4</td>
<td>61</td>
<td>6.4</td>
<td>214.6</td>
<td><span class="tooltip" title="312.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.24</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0.97</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Ember_Spirit" title="Ember Spirit"><img alt="Ember Spirit minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/fc/Ember_Spirit_minimap_icon.png/20px-Ember_Spirit_minimap_icon.png?version=a5ecd78ff88eb8bec8f2b1746195f862" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/fc/Ember_Spirit_minimap_icon.png/30px-Ember_Spirit_minimap_icon.png?version=a5ecd78ff88eb8bec8f2b1746195f862 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/f/fc/Ember_Spirit_minimap_icon.png?version=a5ecd78ff88eb8bec8f2b1746195f862 2x"></a> <a href="/Ember_Spirit" title="Ember Spirit">Ember Spirit</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>21</td>
<td>2.6</td>
<td>83.4</td>
<td>22</td>
<td>2.6</td>
<td>84.4</td>
<td>20</td>
<td>1.8</td>
<td>63.2</td>
<td>63</td>
<td>7</td>
<td>231</td>
<td><span class="tooltip" title="308.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.52</span></td>
<td><span class="tooltip" title="33 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">55</span></td>
<td><span class="tooltip" title="37 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Enchantress" title="Enchantress"><img alt="Enchantress minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/ef/Enchantress_minimap_icon.png/20px-Enchantress_minimap_icon.png?version=ee68064337270e9ff0f900109552e531" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/ef/Enchantress_minimap_icon.png/30px-Enchantress_minimap_icon.png?version=ee68064337270e9ff0f900109552e531 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/e/ef/Enchantress_minimap_icon.png?version=ee68064337270e9ff0f900109552e531 2x"></a> <a href="/Enchantress" title="Enchantress">Enchantress</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>17</td>
<td>1.7</td>
<td>57.8</td>
<td>19</td>
<td>1.8</td>
<td>62.2</td>
<td>22</td>
<td>3.6</td>
<td>108.4</td>
<td>58</td>
<td>7.1</td>
<td>228.4</td>
<td><span class="tooltip" title="323 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">320</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.04</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">55</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">575</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>4</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Enigma" title="Enigma"><img alt="Enigma minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/03/Enigma_minimap_icon.png/20px-Enigma_minimap_icon.png?version=14bbe23667f09ae913d96c5148a492ab" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/03/Enigma_minimap_icon.png/30px-Enigma_minimap_icon.png?version=14bbe23667f09ae913d96c5148a492ab 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/03/Enigma_minimap_icon.png?version=14bbe23667f09ae913d96c5148a492ab 2x"></a> <a href="/Enigma" title="Enigma">Enigma</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2.5</td>
<td>81</td>
<td>14</td>
<td>1</td>
<td>38</td>
<td>16</td>
<td>3.6</td>
<td>102.4</td>
<td>51</td>
<td>7.1</td>
<td>221.4</td>
<td><span class="tooltip" title="292 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.24</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">40</span></td>
<td><span class="tooltip" title="30 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.77</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Faceless_Void" title="Faceless Void"><img alt="Faceless Void minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8e/Faceless_Void_minimap_icon.png/20px-Faceless_Void_minimap_icon.png?version=5d5fa7d2602b006f7c43458335858383" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8e/Faceless_Void_minimap_icon.png/30px-Faceless_Void_minimap_icon.png?version=5d5fa7d2602b006f7c43458335858383 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/8/8e/Faceless_Void_minimap_icon.png?version=5d5fa7d2602b006f7c43458335858383 2x"></a> <a href="/Faceless_Void" title="Faceless Void">Faceless Void</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>24</td>
<td>2.4</td>
<td>81.6</td>
<td>23</td>
<td>3</td>
<td>95</td>
<td>15</td>
<td>1.5</td>
<td>51</td>
<td>62</td>
<td>6.9</td>
<td>227.6</td>
<td><span class="tooltip" title="303.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.68</span></td>
<td><span class="tooltip" title="33 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="39 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">62</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.56</td>
<td>1800</td>
<td>800</td>
<td>1</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Grimstroke" title="Grimstroke"><img alt="Grimstroke minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/bd/Grimstroke_minimap_icon.png/20px-Grimstroke_minimap_icon.png?version=00b023c28171c007a629d8d4778940f2" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/bd/Grimstroke_minimap_icon.png/30px-Grimstroke_minimap_icon.png?version=00b023c28171c007a629d8d4778940f2 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/bd/Grimstroke_minimap_icon.png?version=00b023c28171c007a629d8d4778940f2 2x"></a> <a href="/Grimstroke" title="Grimstroke">Grimstroke</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2.4</td>
<td>78.6</td>
<td>18</td>
<td>1.9</td>
<td>63.6</td>
<td>23</td>
<td>3.8</td>
<td>114.2</td>
<td>62</td>
<td>8.1</td>
<td>256.4</td>
<td><span class="tooltip" title="292.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.88</span></td>
<td><span class="tooltip" title="21 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="25 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">48</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Gyrocopter" title="Gyrocopter"><img alt="Gyrocopter minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/f8/Gyrocopter_minimap_icon.png/20px-Gyrocopter_minimap_icon.png?version=e5add2e7728dd53e7544c207c1824d86" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/f8/Gyrocopter_minimap_icon.png/30px-Gyrocopter_minimap_icon.png?version=e5add2e7728dd53e7544c207c1824d86 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/f/f8/Gyrocopter_minimap_icon.png?version=e5add2e7728dd53e7544c207c1824d86 2x"></a> <a href="/Gyrocopter" title="Gyrocopter">Gyrocopter</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>22</td>
<td>2.3</td>
<td>77.2</td>
<td>19</td>
<td>3.6</td>
<td>105.4</td>
<td>19</td>
<td>2.1</td>
<td>69.4</td>
<td>60</td>
<td>8</td>
<td>252</td>
<td><span class="tooltip" title="318 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">315</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.04</span></td>
<td><span class="tooltip" title="21 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">40</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="3000 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">365</span></td>
<td>125</td>
<td>1.7</td>
<td>0.2</td>
<td>0.97</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Huskar" title="Huskar"><img alt="Huskar minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5b/Huskar_minimap_icon.png/20px-Huskar_minimap_icon.png?version=c5f71860edd2bf9cbf0c335c972e22aa" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5b/Huskar_minimap_icon.png/30px-Huskar_minimap_icon.png?version=c5f71860edd2bf9cbf0c335c972e22aa 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/5b/Huskar_minimap_icon.png?version=c5f71860edd2bf9cbf0c335c972e22aa 2x"></a> <a href="/Huskar" title="Huskar">Huskar</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>21</td>
<td>3.4</td>
<td>102.6</td>
<td>15</td>
<td>1.4</td>
<td>48.6</td>
<td>18</td>
<td>1.5</td>
<td>54</td>
<td>54</td>
<td>6.3</td>
<td>205.2</td>
<td><span class="tooltip" title="292.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.4</span></td>
<td><span class="tooltip" title="19 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">40</span></td>
<td><span class="tooltip" title="24 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="1400 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">400</span></td>
<td>100</td>
<td>1.6</td>
<td>0.3</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Invoker" title="Invoker"><img alt="Invoker minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7d/Invoker_minimap_icon.png/20px-Invoker_minimap_icon.png?version=7b67f7d3885917c3a2f9d5412da762bd" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7d/Invoker_minimap_icon.png/30px-Invoker_minimap_icon.png?version=7b67f7d3885917c3a2f9d5412da762bd 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/7d/Invoker_minimap_icon.png?version=7b67f7d3885917c3a2f9d5412da762bd 2x"></a> <a href="/Invoker" title="Invoker">Invoker</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.4</td>
<td>75.6</td>
<td>14</td>
<td>1.9</td>
<td>59.6</td>
<td>15</td>
<td>4.6</td>
<td>125.4</td>
<td>47</td>
<td>8.9</td>
<td>260.6</td>
<td><span class="tooltip" title="282 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.24</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">48</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Io" title="Io"><img alt="Io minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/86/Io_minimap_icon.png/20px-Io_minimap_icon.png?version=f646c03b9d48eb37c10a3366fb1e4b2a" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/86/Io_minimap_icon.png/30px-Io_minimap_icon.png?version=f646c03b9d48eb37c10a3366fb1e4b2a 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/8/86/Io_minimap_icon.png?version=f646c03b9d48eb37c10a3366fb1e4b2a 2x"></a> <a href="/Io" title="Io">Io</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>17</td>
<td>3</td>
<td>89</td>
<td>14</td>
<td>1.6</td>
<td>52.4</td>
<td>23</td>
<td>1.7</td>
<td>63.8</td>
<td>54</td>
<td>6.3</td>
<td>205.2</td>
<td><span class="tooltip" title="282 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.24</span></td>
<td><span class="tooltip" title="22 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">39</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">48</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.15</td>
<td>0.4</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Jakiro" title="Jakiro"><img alt="Jakiro minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b6/Jakiro_minimap_icon.png/20px-Jakiro_minimap_icon.png?version=f4884e6d627b026437dcd38ea4589c2b" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b6/Jakiro_minimap_icon.png/30px-Jakiro_minimap_icon.png?version=f4884e6d627b026437dcd38ea4589c2b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/b6/Jakiro_minimap_icon.png?version=f4884e6d627b026437dcd38ea4589c2b 2x"></a> <a href="/Jakiro" title="Jakiro">Jakiro</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>27</td>
<td>3</td>
<td>99</td>
<td>10</td>
<td>1.2</td>
<td>38.8</td>
<td>26</td>
<td>3.2</td>
<td>102.8</td>
<td>63</td>
<td>7.4</td>
<td>240.6</td>
<td><span class="tooltip" title="291.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.6</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="1100 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">400</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Juggernaut" title="Juggernaut"><img alt="Juggernaut minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b2/Juggernaut_minimap_icon.png/20px-Juggernaut_minimap_icon.png?version=6919c3a742b3a3e2f0976f9cef472e6b" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b2/Juggernaut_minimap_icon.png/30px-Juggernaut_minimap_icon.png?version=6919c3a742b3a3e2f0976f9cef472e6b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/b2/Juggernaut_minimap_icon.png?version=6919c3a742b3a3e2f0976f9cef472e6b 2x"></a> <a href="/Juggernaut" title="Juggernaut">Juggernaut</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>20</td>
<td>2.2</td>
<td>72.8</td>
<td>34</td>
<td>2.8</td>
<td>101.2</td>
<td>14</td>
<td>1.4</td>
<td>47.6</td>
<td>68</td>
<td>6.4</td>
<td>221.6</td>
<td><span class="tooltip" title="305.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.44</span></td>
<td><span class="tooltip" title="12 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="16 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>110</td>
<td>1.4</td>
<td>0.33</td>
<td>0.84</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Keeper_of_the_Light" title="Keeper of the Light"><img alt="Keeper of the Light minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/76/Keeper_of_the_Light_minimap_icon.png/20px-Keeper_of_the_Light_minimap_icon.png?version=d58969f2955c7958a1a3c9e6a9125eaf" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/76/Keeper_of_the_Light_minimap_icon.png/30px-Keeper_of_the_Light_minimap_icon.png?version=d58969f2955c7958a1a3c9e6a9125eaf 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/76/Keeper_of_the_Light_minimap_icon.png?version=d58969f2955c7958a1a3c9e6a9125eaf 2x"></a> <a href="/Keeper_of_the_Light" title="Keeper of the Light">Keeper of the Light</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>16</td>
<td>2.3</td>
<td>71.2</td>
<td>15</td>
<td>1.6</td>
<td>53.4</td>
<td>23</td>
<td>3.2</td>
<td>99.8</td>
<td>54</td>
<td>7.1</td>
<td>224.4</td>
<td><span class="tooltip" title="332.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">330</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.4</span></td>
<td><span class="tooltip" title="20 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">43</span></td>
<td><span class="tooltip" title="27 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.85</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Kunkka" title="Kunkka"><img alt="Kunkka minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5b/Kunkka_minimap_icon.png/20px-Kunkka_minimap_icon.png?version=403c09ec9a808d6a8a4f1507783adb83" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/5b/Kunkka_minimap_icon.png/30px-Kunkka_minimap_icon.png?version=403c09ec9a808d6a8a4f1507783adb83 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/5b/Kunkka_minimap_icon.png?version=403c09ec9a808d6a8a4f1507783adb83 2x"></a> <a href="/Kunkka" title="Kunkka">Kunkka</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>24</td>
<td>3.8</td>
<td>115.2</td>
<td>14</td>
<td>1.3</td>
<td>45.2</td>
<td>18</td>
<td>1.5</td>
<td>54</td>
<td>56</td>
<td>6.6</td>
<td>214.4</td>
<td><span class="tooltip" title="302.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.24</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Legion_Commander" title="Legion Commander"><img alt="Legion Commander minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/10/Legion_Commander_minimap_icon.png/20px-Legion_Commander_minimap_icon.png?version=53787dfff740c52e9b827091b7f59f92" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/10/Legion_Commander_minimap_icon.png/30px-Legion_Commander_minimap_icon.png?version=53787dfff740c52e9b827091b7f59f92 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/10/Legion_Commander_minimap_icon.png?version=53787dfff740c52e9b827091b7f59f92 2x"></a> <a href="/Legion_Commander" title="Legion Commander">Legion Commander</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>26</td>
<td>3.3</td>
<td>105.2</td>
<td>18</td>
<td>1.7</td>
<td>58.8</td>
<td>20</td>
<td>2.2</td>
<td>72.8</td>
<td>64</td>
<td>7.2</td>
<td>236.8</td>
<td><span class="tooltip" title="333 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">330</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.88</span></td>
<td><span class="tooltip" title="35 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="39 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">65</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.46</td>
<td>0.64</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Leshrac" title="Leshrac"><img alt="Leshrac minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/c8/Leshrac_minimap_icon.png/20px-Leshrac_minimap_icon.png?version=268ab9030a231fa8b73588bc237e6c45" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/c8/Leshrac_minimap_icon.png/30px-Leshrac_minimap_icon.png?version=268ab9030a231fa8b73588bc237e6c45 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/c/c8/Leshrac_minimap_icon.png?version=268ab9030a231fa8b73588bc237e6c45 2x"></a> <a href="/Leshrac" title="Leshrac">Leshrac</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.3</td>
<td>73.2</td>
<td>23</td>
<td>2.3</td>
<td>78.2</td>
<td>22</td>
<td>3.5</td>
<td>106</td>
<td>63</td>
<td>8.1</td>
<td>257.4</td>
<td><span class="tooltip" title="328.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">325</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.68</span></td>
<td><span class="tooltip" title="19 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">41</span></td>
<td><span class="tooltip" title="23 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.77</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>4</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Lich" title="Lich"><img alt="Lich minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/08/Lich_minimap_icon.png/20px-Lich_minimap_icon.png?version=cd857b62405ac8247a300a5a6b75b51c" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/08/Lich_minimap_icon.png/30px-Lich_minimap_icon.png?version=cd857b62405ac8247a300a5a6b75b51c 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/08/Lich_minimap_icon.png?version=cd857b62405ac8247a300a5a6b75b51c 2x"></a> <a href="/Lich" title="Lich">Lich</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>20</td>
<td>2.1</td>
<td>70.4</td>
<td>15</td>
<td>2</td>
<td>63</td>
<td>24</td>
<td>4.1</td>
<td>122.4</td>
<td>59</td>
<td>8.2</td>
<td>255.8</td>
<td><span class="tooltip" title="297.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.4</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.7</td>
<td>0.46</td>
<td>0.54</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Lifestealer" title="Lifestealer"><img alt="Lifestealer minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/74/Lifestealer_minimap_icon.png/20px-Lifestealer_minimap_icon.png?version=8e8e9bd0d91dc87d1c64d1a28ba5be9f" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/74/Lifestealer_minimap_icon.png/30px-Lifestealer_minimap_icon.png?version=8e8e9bd0d91dc87d1c64d1a28ba5be9f 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/74/Lifestealer_minimap_icon.png?version=8e8e9bd0d91dc87d1c64d1a28ba5be9f 2x"></a> <a href="/Lifestealer" title="Lifestealer">Lifestealer</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>25</td>
<td>2.4</td>
<td>82.6</td>
<td>18</td>
<td>2.4</td>
<td>75.6</td>
<td>15</td>
<td>1.8</td>
<td>58.2</td>
<td>58</td>
<td>6.6</td>
<td>216.4</td>
<td><span class="tooltip" title="327.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">325</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.88</span></td>
<td><span class="tooltip" title="20 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="30 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">55</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>140</td>
<td>1.7</td>
<td>0.39</td>
<td>0.44</td>
<td>1800</td>
<td>800</td>
<td>1</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Lina" title="Lina"><img alt="Lina minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/44/Lina_minimap_icon.png/20px-Lina_minimap_icon.png?version=ef438ae8e520cda94333907c08da9ae5" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/44/Lina_minimap_icon.png/30px-Lina_minimap_icon.png?version=ef438ae8e520cda94333907c08da9ae5 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/4/44/Lina_minimap_icon.png?version=ef438ae8e520cda94333907c08da9ae5 2x"></a> <a href="/Lina" title="Lina">Lina</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>20</td>
<td>2.4</td>
<td>77.6</td>
<td>23</td>
<td>1.8</td>
<td>66.2</td>
<td>30</td>
<td>3.7</td>
<td>118.8</td>
<td>73</td>
<td>7.9</td>
<td>262.6</td>
<td><span class="tooltip" title="293.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.68</span></td>
<td><span class="tooltip" title="21 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="1000 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">670</span></td>
<td>100</td>
<td>1.6</td>
<td>0.75</td>
<td>0.78</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Lion" title="Lion"><img alt="Lion minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/13/Lion_minimap_icon.png/20px-Lion_minimap_icon.png?version=210b9a5da3bd35012c2bb972cee3f7f4" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/13/Lion_minimap_icon.png/30px-Lion_minimap_icon.png?version=210b9a5da3bd35012c2bb972cee3f7f4 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/13/Lion_minimap_icon.png?version=210b9a5da3bd35012c2bb972cee3f7f4 2x"></a> <a href="/Lion" title="Lion">Lion</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.2</td>
<td>70.8</td>
<td>15</td>
<td>1.5</td>
<td>51</td>
<td>18</td>
<td>3.5</td>
<td>102</td>
<td>51</td>
<td>7.2</td>
<td>223.8</td>
<td><span class="tooltip" title="292.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.4</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.43</td>
<td>0.74</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Lone_Druid" title="Lone Druid"><img alt="Lone Druid minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/08/Lone_Druid_minimap_icon.png/20px-Lone_Druid_minimap_icon.png?version=6e0f14a32208de100f62c3cb7d9e21cc" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/08/Lone_Druid_minimap_icon.png/30px-Lone_Druid_minimap_icon.png?version=6e0f14a32208de100f62c3cb7d9e21cc 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/08/Lone_Druid_minimap_icon.png?version=6e0f14a32208de100f62c3cb7d9e21cc 2x"></a> <a href="/Lone_Druid" title="Lone Druid">Lone Druid</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>18</td>
<td>2.7</td>
<td>82.8</td>
<td>20</td>
<td>2.8</td>
<td>87.2</td>
<td>13</td>
<td>1.4</td>
<td>46.6</td>
<td>51</td>
<td>6.9</td>
<td>216.6</td>
<td><span class="tooltip" title="328.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">325</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.2</span></td>
<td><span class="tooltip" title="18 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">38</span></td>
<td><span class="tooltip" title="22 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.7</td>
<td>0.33</td>
<td>0.53</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Luna" title="Luna"><img alt="Luna minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8c/Luna_minimap_icon.png/20px-Luna_minimap_icon.png?version=066101cfb41423fb3296d41cac7a7fca" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8c/Luna_minimap_icon.png/30px-Luna_minimap_icon.png?version=066101cfb41423fb3296d41cac7a7fca 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/8/8c/Luna_minimap_icon.png?version=066101cfb41423fb3296d41cac7a7fca 2x"></a> <a href="/Luna" title="Luna">Luna</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>18</td>
<td>2.2</td>
<td>70.8</td>
<td>18</td>
<td>3.6</td>
<td>104.4</td>
<td>18</td>
<td>1.9</td>
<td>63.6</td>
<td>54</td>
<td>7.7</td>
<td>238.8</td>
<td><span class="tooltip" title="327.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">325</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.88</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="32 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">330</span></td>
<td>100</td>
<td>1.7</td>
<td>0.46</td>
<td>0.54</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Lycan" title="Lycan"><img alt="Lycan minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/4f/Lycan_minimap_icon.png/20px-Lycan_minimap_icon.png?version=97d1843469d941e9a631326116c97bc6" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/4f/Lycan_minimap_icon.png/30px-Lycan_minimap_icon.png?version=97d1843469d941e9a631326116c97bc6 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/4/4f/Lycan_minimap_icon.png?version=97d1843469d941e9a631326116c97bc6 2x"></a> <a href="/Lycan" title="Lycan">Lycan</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>26</td>
<td>3.4</td>
<td>107.6</td>
<td>16</td>
<td>1</td>
<td>40</td>
<td>19</td>
<td>1.4</td>
<td>52.6</td>
<td>61</td>
<td>5.8</td>
<td>200.2</td>
<td><span class="tooltip" title="317.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">315</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.56</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.55</td>
<td>0.55</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>8</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Magnus" title="Magnus"><img alt="Magnus minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7f/Magnus_minimap_icon.png/20px-Magnus_minimap_icon.png?version=ae65dabe1e44f07f7224d9bbbfaae7b0" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7f/Magnus_minimap_icon.png/30px-Magnus_minimap_icon.png?version=ae65dabe1e44f07f7224d9bbbfaae7b0 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/7f/Magnus_minimap_icon.png?version=ae65dabe1e44f07f7224d9bbbfaae7b0 2x"></a> <a href="/Magnus" title="Magnus">Magnus</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>25</td>
<td>3.5</td>
<td>109</td>
<td>15</td>
<td>2.5</td>
<td>75</td>
<td>19</td>
<td>1.7</td>
<td>59.8</td>
<td>59</td>
<td>7.7</td>
<td>243.8</td>
<td><span class="tooltip" title="307.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.4</span></td>
<td><span class="tooltip" title="32 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="44 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">69</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.8</td>
<td>0.5</td>
<td>0.84</td>
<td>1800</td>
<td>800</td>
<td>0.8</td>
<td>24</td>
<td>0.5</td>
<td>4</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Mars" title="Mars"><img alt="Mars minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/6c/Mars_minimap_icon.png/20px-Mars_minimap_icon.png?version=afeadec52a7d04a2d76256738bc56a15" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/6c/Mars_minimap_icon.png/30px-Mars_minimap_icon.png?version=afeadec52a7d04a2d76256738bc56a15 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/6c/Mars_minimap_icon.png?version=afeadec52a7d04a2d76256738bc56a15 2x"></a> <a href="/Mars" title="Mars">Mars</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>3.2</td>
<td>99.8</td>
<td>20</td>
<td>1.9</td>
<td>65.6</td>
<td>17</td>
<td>1.4</td>
<td>50.6</td>
<td>60</td>
<td>6.5</td>
<td>216</td>
<td><span class="tooltip" title="313.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.2</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">52</span></td>
<td><span class="tooltip" title="37 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">250</span></td>
<td>100</td>
<td>1.8</td>
<td>0.4</td>
<td>0</td>
<td>1800</td>
<td>800</td>
<td>0.8</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Medusa" title="Medusa"><img alt="Medusa minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/64/Medusa_minimap_icon.png/20px-Medusa_minimap_icon.png?version=e469755af3cd11ffb7b12282b0c60c75" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/64/Medusa_minimap_icon.png/30px-Medusa_minimap_icon.png?version=e469755af3cd11ffb7b12282b0c60c75 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/64/Medusa_minimap_icon.png?version=e469755af3cd11ffb7b12282b0c60c75 2x"></a> <a href="/Medusa" title="Medusa">Medusa</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>15</td>
<td>1.5</td>
<td>51</td>
<td>22</td>
<td>3.6</td>
<td>108.4</td>
<td>19</td>
<td>3.4</td>
<td>100.6</td>
<td>56</td>
<td>8.5</td>
<td>260</td>
<td><span class="tooltip" title="278 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">275</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.52</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="30 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">52</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.6</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Meepo" title="Meepo"><img alt="Meepo minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/17/Meepo_minimap_icon.png/20px-Meepo_minimap_icon.png?version=67827e8e398d221dee25e68461319d54" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/17/Meepo_minimap_icon.png/30px-Meepo_minimap_icon.png?version=67827e8e398d221dee25e68461319d54 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/17/Meepo_minimap_icon.png?version=67827e8e398d221dee25e68461319d54 2x"></a> <a href="/Meepo" title="Meepo">Meepo</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>24</td>
<td>1.8</td>
<td>67.2</td>
<td>24</td>
<td>1.8</td>
<td>67.2</td>
<td>20</td>
<td>1.6</td>
<td>58.4</td>
<td>68</td>
<td>5.2</td>
<td>192.8</td>
<td><span class="tooltip" title="334 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">330</span></td>
<td><span class="tooltip" title="3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">6.84</span></td>
<td><span class="tooltip" title="22 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="28 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">52</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.38</td>
<td>0.6</td>
<td>1800</td>
<td>800</td>
<td>0.65</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Mirana" title="Mirana"><img alt="Mirana minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b3/Mirana_minimap_icon.png/20px-Mirana_minimap_icon.png?version=4c72f92657861e93f24f06bb8eabb268" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b3/Mirana_minimap_icon.png/30px-Mirana_minimap_icon.png?version=4c72f92657861e93f24f06bb8eabb268 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/b3/Mirana_minimap_icon.png?version=4c72f92657861e93f24f06bb8eabb268 2x"></a> <a href="/Mirana" title="Mirana">Mirana</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>18</td>
<td>2.2</td>
<td>70.8</td>
<td>18</td>
<td>3.7</td>
<td>106.8</td>
<td>22</td>
<td>1.9</td>
<td>67.6</td>
<td>58</td>
<td>7.8</td>
<td>245.2</td>
<td><span class="tooltip" title="292.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.88</span></td>
<td><span class="tooltip" title="25 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">43</span></td>
<td><span class="tooltip" title="30 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">48</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">630</span></td>
<td>115</td>
<td>1.7</td>
<td>0.35</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Monkey_King" title="Monkey King"><img alt="Monkey King minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/76/Monkey_King_minimap_icon.png/20px-Monkey_King_minimap_icon.png?version=4a6bee25f6d11db01ba22bf70ad79730" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/76/Monkey_King_minimap_icon.png/30px-Monkey_King_minimap_icon.png?version=4a6bee25f6d11db01ba22bf70ad79730 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/76/Monkey_King_minimap_icon.png?version=4a6bee25f6d11db01ba22bf70ad79730 2x"></a> <a href="/Monkey_King" title="Monkey King">Monkey King</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>19</td>
<td>2.8</td>
<td>86.2</td>
<td>22</td>
<td>3.7</td>
<td>110.8</td>
<td>20</td>
<td>1.8</td>
<td>63.2</td>
<td>61</td>
<td>8.3</td>
<td>260.2</td>
<td><span class="tooltip" title="308.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.52</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td>100</td>
<td>1.7</td>
<td>0.45</td>
<td>0.2</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>8</td>
<td>1.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Morphling" title="Morphling"><img alt="Morphling minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/13/Morphling_minimap_icon.png/20px-Morphling_minimap_icon.png?version=cbaf89f0d44f21647134ea2bb443f144" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/13/Morphling_minimap_icon.png/30px-Morphling_minimap_icon.png?version=cbaf89f0d44f21647134ea2bb443f144 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/13/Morphling_minimap_icon.png?version=cbaf89f0d44f21647134ea2bb443f144 2x"></a> <a href="/Morphling" title="Morphling">Morphling</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>22</td>
<td>3</td>
<td>94</td>
<td>24</td>
<td>4.3</td>
<td>127.2</td>
<td>15</td>
<td>1.5</td>
<td>51</td>
<td>61</td>
<td>8.8</td>
<td>272.2</td>
<td><span class="tooltip" title="283.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.84</span></td>
<td><span class="tooltip" title="9 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">33</span></td>
<td><span class="tooltip" title="18 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="1300 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">350</span></td>
<td>100</td>
<td>1.5</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Naga_Siren" title="Naga Siren"><img alt="Naga Siren minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/6c/Naga_Siren_minimap_icon.png/20px-Naga_Siren_minimap_icon.png?version=cb686daa5cf455927843b1879a3278bd" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/6c/Naga_Siren_minimap_icon.png/30px-Naga_Siren_minimap_icon.png?version=cb686daa5cf455927843b1879a3278bd 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/6c/Naga_Siren_minimap_icon.png?version=cb686daa5cf455927843b1879a3278bd 2x"></a> <a href="/Naga_Siren" title="Naga Siren">Naga Siren</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>22</td>
<td>2.8</td>
<td>89.2</td>
<td>21</td>
<td>3.5</td>
<td>105</td>
<td>21</td>
<td>2</td>
<td>69</td>
<td>64</td>
<td>8.3</td>
<td>263.2</td>
<td><span class="tooltip" title="323.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">320</span></td>
<td><span class="tooltip" title="3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">6.36</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="25 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>8</td>
<td>1.5</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Nature%27s_Prophet" title="Natures Prophet"><img alt="Natures Prophet minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a4/Nature%27s_Prophet_minimap_icon.png/20px-Nature%27s_Prophet_minimap_icon.png?version=2f3e118ef524adefd94c4663df5b207c" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a4/Nature%27s_Prophet_minimap_icon.png/30px-Nature%27s_Prophet_minimap_icon.png?version=2f3e118ef524adefd94c4663df5b207c 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/a/a4/Nature%27s_Prophet_minimap_icon.png?version=2f3e118ef524adefd94c4663df5b207c 2x"></a> <a href="/Nature%27s_Prophet" title="Natures Prophet">Natures Prophet</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2.6</td>
<td>83.4</td>
<td>22</td>
<td>3.6</td>
<td>108.4</td>
<td>23</td>
<td>3.7</td>
<td>111.8</td>
<td>66</td>
<td>9.9</td>
<td>303.6</td>
<td><span class="tooltip" title="293.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.52</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="37 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="1125 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.77</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Necrophos" title="Necrophos"><img alt="Necrophos minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/af/Necrophos_minimap_icon.png/20px-Necrophos_minimap_icon.png?version=52c8b0e3aefd5e76b066464e1ae71b25" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/af/Necrophos_minimap_icon.png/30px-Necrophos_minimap_icon.png?version=52c8b0e3aefd5e76b066464e1ae71b25 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/a/af/Necrophos_minimap_icon.png?version=52c8b0e3aefd5e76b066464e1ae71b25 2x"></a> <a href="/Necrophos" title="Necrophos">Necrophos</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.8</td>
<td>85.2</td>
<td>12</td>
<td>1.3</td>
<td>43.2</td>
<td>21</td>
<td>2.9</td>
<td>90.6</td>
<td>51</td>
<td>7</td>
<td>219</td>
<td><span class="tooltip" title="281.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.92</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="30 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.47</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Night_Stalker" title="Night Stalker"><img alt="Night Stalker minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7f/Night_Stalker_minimap_icon.png/20px-Night_Stalker_minimap_icon.png?version=0954e5e2919ec9793af972fa8d2dc33a" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7f/Night_Stalker_minimap_icon.png/30px-Night_Stalker_minimap_icon.png?version=0954e5e2919ec9793af972fa8d2dc33a 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/7f/Night_Stalker_minimap_icon.png?version=0954e5e2919ec9793af972fa8d2dc33a 2x"></a> <a href="/Night_Stalker" title="Night Stalker">Night Stalker</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>3.2</td>
<td>99.8</td>
<td>18</td>
<td>2.3</td>
<td>73.2</td>
<td>13</td>
<td>1.6</td>
<td>51.4</td>
<td>54</td>
<td>7.1</td>
<td>224.4</td>
<td><span class="tooltip" title="297.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.88</span></td>
<td><span class="tooltip" title="38 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="42 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">65</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.55</td>
<td>0.55</td>
<td>800</td>
<td>1800</td>
<td>0.5</td>
<td>24</td>
<td>1.75</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Nyx_Assassin" title="Nyx Assassin"><img alt="Nyx Assassin minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/aa/Nyx_Assassin_minimap_icon.png/20px-Nyx_Assassin_minimap_icon.png?version=ba3b62dadb9c933b28f2dd8ce4f7e86a" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/aa/Nyx_Assassin_minimap_icon.png/30px-Nyx_Assassin_minimap_icon.png?version=ba3b62dadb9c933b28f2dd8ce4f7e86a 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/a/aa/Nyx_Assassin_minimap_icon.png?version=ba3b62dadb9c933b28f2dd8ce4f7e86a 2x"></a> <a href="/Nyx_Assassin" title="Nyx Assassin">Nyx Assassin</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>18</td>
<td>2.5</td>
<td>78</td>
<td>19</td>
<td>2.5</td>
<td>79</td>
<td>18</td>
<td>2.1</td>
<td>68.4</td>
<td>55</td>
<td>7.1</td>
<td>225.4</td>
<td><span class="tooltip" title="318 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">315</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.04</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.46</td>
<td>0.54</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>2.5</td>
<td>6</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Ogre_Magi" title="Ogre Magi"><img alt="Ogre Magi minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/55/Ogre_Magi_minimap_icon.png/20px-Ogre_Magi_minimap_icon.png?version=7770fb9997efe35adcbeded53d96e120" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/55/Ogre_Magi_minimap_icon.png/30px-Ogre_Magi_minimap_icon.png?version=7770fb9997efe35adcbeded53d96e120 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/55/Ogre_Magi_minimap_icon.png?version=7770fb9997efe35adcbeded53d96e120 2x"></a> <a href="/Ogre_Magi" title="Ogre Magi">Ogre Magi</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>23</td>
<td>3.5</td>
<td>107</td>
<td>14</td>
<td>1.9</td>
<td>59.6</td>
<td>15</td>
<td>2.5</td>
<td>75</td>
<td>52</td>
<td>7.9</td>
<td>241.6</td>
<td><span class="tooltip" title="292 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="6 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">8.24</span></td>
<td><span class="tooltip" title="39 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="45 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>3.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Omniknight" title="Omniknight"><img alt="Omniknight minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/05/Omniknight_minimap_icon.png/20px-Omniknight_minimap_icon.png?version=b3c2816bb5b20f9e9ba8fef904eb3b80" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/05/Omniknight_minimap_icon.png/30px-Omniknight_minimap_icon.png?version=b3c2816bb5b20f9e9ba8fef904eb3b80 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/05/Omniknight_minimap_icon.png?version=b3c2816bb5b20f9e9ba8fef904eb3b80 2x"></a> <a href="/Omniknight" title="Omniknight">Omniknight</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>3.6</td>
<td>108.4</td>
<td>15</td>
<td>1.8</td>
<td>58.2</td>
<td>15</td>
<td>1.8</td>
<td>58.2</td>
<td>52</td>
<td>7.2</td>
<td>224.8</td>
<td><span class="tooltip" title="302.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.4</span></td>
<td><span class="tooltip" title="31 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.433</td>
<td>0.567</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Oracle" title="Oracle"><img alt="Oracle minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/61/Oracle_minimap_icon.png/20px-Oracle_minimap_icon.png?version=8565cf936fa705c0fb53a00410812a1b" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/61/Oracle_minimap_icon.png/30px-Oracle_minimap_icon.png?version=8565cf936fa705c0fb53a00410812a1b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/61/Oracle_minimap_icon.png?version=8565cf936fa705c0fb53a00410812a1b 2x"></a> <a href="/Oracle" title="Oracle">Oracle</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>20</td>
<td>2.4</td>
<td>77.6</td>
<td>15</td>
<td>1.7</td>
<td>55.8</td>
<td>26</td>
<td>4</td>
<td>122</td>
<td>61</td>
<td>8.1</td>
<td>255.4</td>
<td><span class="tooltip" title="297.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.4</span></td>
<td><span class="tooltip" title="13 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">39</span></td>
<td><span class="tooltip" title="19 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">620</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Outworld_Devourer" title="Outworld Devourer"><img alt="Outworld Devourer minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/28/Outworld_Devourer_minimap_icon.png/20px-Outworld_Devourer_minimap_icon.png?version=5ea6a05053dfb97c247a6ec8956ae317" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/28/Outworld_Devourer_minimap_icon.png/30px-Outworld_Devourer_minimap_icon.png?version=5ea6a05053dfb97c247a6ec8956ae317 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/28/Outworld_Devourer_minimap_icon.png?version=5ea6a05053dfb97c247a6ec8956ae317 2x"></a> <a href="/Outworld_Devourer" title="Outworld Devourer">Outworld Devourer</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2.9</td>
<td>90.6</td>
<td>24</td>
<td>2</td>
<td>72</td>
<td>30</td>
<td>4.2</td>
<td>130.8</td>
<td>75</td>
<td>9.1</td>
<td>293.4</td>
<td><span class="tooltip" title="313.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="1.5 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.34</span></td>
<td><span class="tooltip" title="16 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">450</span></td>
<td>100</td>
<td>1.9</td>
<td>0.46</td>
<td>0.54</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>4</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Pangolier" title="Pangolier"><img alt="Pangolier minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/80/Pangolier_minimap_icon.png/20px-Pangolier_minimap_icon.png?version=150813f0db9517b1c0e2081a1915d9d2" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/80/Pangolier_minimap_icon.png/30px-Pangolier_minimap_icon.png?version=150813f0db9517b1c0e2081a1915d9d2 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/8/80/Pangolier_minimap_icon.png?version=150813f0db9517b1c0e2081a1915d9d2 2x"></a> <a href="/Pangolier" title="Pangolier">Pangolier</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>17</td>
<td>2.5</td>
<td>77</td>
<td>18</td>
<td>3.2</td>
<td>94.8</td>
<td>16</td>
<td>1.9</td>
<td>61.6</td>
<td>51</td>
<td>7.6</td>
<td>233.4</td>
<td><span class="tooltip" title="307.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.88</span></td>
<td><span class="tooltip" title="33 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="39 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.33</td>
<td>0</td>
<td>1800</td>
<td>800</td>
<td>1</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Phantom_Assassin" title="Phantom Assassin"><img alt="Phantom Assassin minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/c9/Phantom_Assassin_minimap_icon.png/20px-Phantom_Assassin_minimap_icon.png?version=fb8dbc3ce3afbbd198d6fa111f864302" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/c9/Phantom_Assassin_minimap_icon.png/30px-Phantom_Assassin_minimap_icon.png?version=fb8dbc3ce3afbbd198d6fa111f864302 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/c/c9/Phantom_Assassin_minimap_icon.png?version=fb8dbc3ce3afbbd198d6fa111f864302 2x"></a> <a href="/Phantom_Assassin" title="Phantom Assassin">Phantom Assassin</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>21</td>
<td>2.2</td>
<td>73.8</td>
<td>23</td>
<td>3.4</td>
<td>104.6</td>
<td>15</td>
<td>1.4</td>
<td>48.6</td>
<td>59</td>
<td>7</td>
<td>227</td>
<td><span class="tooltip" title="308.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.68</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="26 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Phantom_Lancer" title="Phantom Lancer"><img alt="Phantom Lancer minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/f9/Phantom_Lancer_minimap_icon.png/20px-Phantom_Lancer_minimap_icon.png?version=95e13b1bdd0d5da252bb22cb8e7d91cf" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/f9/Phantom_Lancer_minimap_icon.png/30px-Phantom_Lancer_minimap_icon.png?version=95e13b1bdd0d5da252bb22cb8e7d91cf 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/f/f9/Phantom_Lancer_minimap_icon.png?version=95e13b1bdd0d5da252bb22cb8e7d91cf 2x"></a> <a href="/Phantom_Lancer" title="Phantom Lancer">Phantom Lancer</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>19</td>
<td>2.2</td>
<td>71.8</td>
<td>29</td>
<td>3.2</td>
<td>105.8</td>
<td>19</td>
<td>2</td>
<td>67</td>
<td>67</td>
<td>7.4</td>
<td>244.6</td>
<td><span class="tooltip" title="294.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.64</span></td>
<td><span class="tooltip" title="22 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="44 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">73</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>8</td>
<td>1.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Phoenix" title="Phoenix"><img alt="Phoenix minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/21/Phoenix_minimap_icon.png/20px-Phoenix_minimap_icon.png?version=eaf93fb583044d0b2fd451c08bcc36b8" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/21/Phoenix_minimap_icon.png/30px-Phoenix_minimap_icon.png?version=eaf93fb583044d0b2fd451c08bcc36b8 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/21/Phoenix_minimap_icon.png?version=eaf93fb583044d0b2fd451c08bcc36b8 2x"></a> <a href="/Phoenix" title="Phoenix">Phoenix</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>3.7</td>
<td>111.8</td>
<td>12</td>
<td>1.3</td>
<td>43.2</td>
<td>18</td>
<td>1.8</td>
<td>61.2</td>
<td>53</td>
<td>6.8</td>
<td>216.2</td>
<td><span class="tooltip" title="281.7 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0.92</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="1100 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0.633</td>
<td>1800</td>
<td>800</td>
<td>1</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Puck" title="Puck"><img alt="Puck minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/cf/Puck_minimap_icon.png/20px-Puck_minimap_icon.png?version=45cdc499f81dadd8e6f623facf2ece1c" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/cf/Puck_minimap_icon.png/30px-Puck_minimap_icon.png?version=45cdc499f81dadd8e6f623facf2ece1c 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/c/cf/Puck_minimap_icon.png?version=45cdc499f81dadd8e6f623facf2ece1c 2x"></a> <a href="/Puck" title="Puck">Puck</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>17</td>
<td>2.4</td>
<td>74.6</td>
<td>22</td>
<td>2.2</td>
<td>74.8</td>
<td>23</td>
<td>3.5</td>
<td>107</td>
<td>62</td>
<td>8.1</td>
<td>256.4</td>
<td><span class="tooltip" title="293.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0.52</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">64</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.8</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Pudge" title="Pudge"><img alt="Pudge minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/55/Pudge_minimap_icon.png/20px-Pudge_minimap_icon.png?version=27992ab3376de71767571b27f182403f" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/55/Pudge_minimap_icon.png/30px-Pudge_minimap_icon.png?version=27992ab3376de71767571b27f182403f 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/5/55/Pudge_minimap_icon.png?version=27992ab3376de71767571b27f182403f 2x"></a> <a href="/Pudge" title="Pudge">Pudge</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>25</td>
<td>4</td>
<td>121</td>
<td>14</td>
<td>1.5</td>
<td>50</td>
<td>16</td>
<td>1.5</td>
<td>52</td>
<td>55</td>
<td>7</td>
<td>223</td>
<td><span class="tooltip" title="282 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0.24</span></td>
<td><span class="tooltip" title="40 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">65</span></td>
<td><span class="tooltip" title="46 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">71</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>1.17</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Pugna" title="Pugna"><img alt="Pugna minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/ab/Pugna_minimap_icon.png/20px-Pugna_minimap_icon.png?version=8ec6c1073df6e4b992d6d13774d06117" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/ab/Pugna_minimap_icon.png/30px-Pugna_minimap_icon.png?version=8ec6c1073df6e4b992d6d13774d06117 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/a/ab/Pugna_minimap_icon.png?version=8ec6c1073df6e4b992d6d13774d06117 2x"></a> <a href="/Pugna" title="Pugna">Pugna</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>19</td>
<td>2</td>
<td>67</td>
<td>16</td>
<td>1.3</td>
<td>47.2</td>
<td>24</td>
<td>5.2</td>
<td>148.8</td>
<td>59</td>
<td>8.5</td>
<td>263</td>
<td><span class="tooltip" title="332.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">330</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.56</span></td>
<td><span class="tooltip" title="19 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">43</span></td>
<td><span class="tooltip" title="27 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">630</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Queen_of_Pain" title="Queen of Pain"><img alt="Queen of Pain minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0f/Queen_of_Pain_minimap_icon.png/20px-Queen_of_Pain_minimap_icon.png?version=de4f36a4a0e957f54bace07aca1cc58f" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0f/Queen_of_Pain_minimap_icon.png/30px-Queen_of_Pain_minimap_icon.png?version=de4f36a4a0e957f54bace07aca1cc58f 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/0f/Queen_of_Pain_minimap_icon.png?version=de4f36a4a0e957f54bace07aca1cc58f 2x"></a> <a href="/Queen_of_Pain" title="Queen of Pain">Queen of Pain</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.3</td>
<td>73.2</td>
<td>22</td>
<td>2.5</td>
<td>82</td>
<td>25</td>
<td>2.9</td>
<td>94.6</td>
<td>65</td>
<td>7.7</td>
<td>249.8</td>
<td><span class="tooltip" title="293.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.52</span></td>
<td><span class="tooltip" title="20 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="28 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="1500 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.5</td>
<td>0.56</td>
<td>0.41</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Razor" title="Razor"><img alt="Razor minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2b/Razor_minimap_icon.png/20px-Razor_minimap_icon.png?version=b5dd627447b6874fc936bab715d66493" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2b/Razor_minimap_icon.png/30px-Razor_minimap_icon.png?version=b5dd627447b6874fc936bab715d66493 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/2b/Razor_minimap_icon.png?version=b5dd627447b6874fc936bab715d66493 2x"></a> <a href="/Razor" title="Razor">Razor</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>22</td>
<td>2.6</td>
<td>84.4</td>
<td>22</td>
<td>2.1</td>
<td>72.4</td>
<td>21</td>
<td>1.8</td>
<td>64.2</td>
<td>65</td>
<td>6.5</td>
<td>221</td>
<td><span class="tooltip" title="288.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.52</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="25 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="2000 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">475</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.4</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Riki" title="Riki"><img alt="Riki minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a4/Riki_minimap_icon.png/20px-Riki_minimap_icon.png?version=79e7804eb48cdcd259c3dff596dcbd00" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a4/Riki_minimap_icon.png/30px-Riki_minimap_icon.png?version=79e7804eb48cdcd259c3dff596dcbd00 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/a/a4/Riki_minimap_icon.png?version=79e7804eb48cdcd259c3dff596dcbd00 2x"></a> <a href="/Riki" title="Riki">Riki</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>18</td>
<td>2.4</td>
<td>75.6</td>
<td>18</td>
<td>1.4</td>
<td>51.6</td>
<td>14</td>
<td>1.3</td>
<td>45.2</td>
<td>50</td>
<td>5.1</td>
<td>172.4</td>
<td><span class="tooltip" title="322.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">320</span></td>
<td><span class="tooltip" title="3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.88</span></td>
<td><span class="tooltip" title="41 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="45 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>3</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Rubick" title="Rubick"><img alt="Rubick minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/61/Rubick_minimap_icon.png/20px-Rubick_minimap_icon.png?version=0072f8dfa7e1d7b446a227f2f827c713" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/61/Rubick_minimap_icon.png/30px-Rubick_minimap_icon.png?version=0072f8dfa7e1d7b446a227f2f827c713 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/61/Rubick_minimap_icon.png?version=0072f8dfa7e1d7b446a227f2f827c713 2x"></a> <a href="/Rubick" title="Rubick">Rubick</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2</td>
<td>69</td>
<td>23</td>
<td>2.5</td>
<td>83</td>
<td>25</td>
<td>3.1</td>
<td>99.4</td>
<td>69</td>
<td>7.6</td>
<td>251.4</td>
<td><span class="tooltip" title="293.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.68</span></td>
<td><span class="tooltip" title="22 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="32 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="1125 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.77</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Sand_King" title="Sand King"><img alt="Sand King minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9f/Sand_King_minimap_icon.png/20px-Sand_King_minimap_icon.png?version=8bc7816073555fe2e381c8ad851d848b" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/9f/Sand_King_minimap_icon.png/30px-Sand_King_minimap_icon.png?version=8bc7816073555fe2e381c8ad851d848b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/9f/Sand_King_minimap_icon.png?version=8bc7816073555fe2e381c8ad851d848b 2x"></a> <a href="/Sand_King" title="Sand King">Sand King</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>3</td>
<td>94</td>
<td>19</td>
<td>1.8</td>
<td>62.2</td>
<td>19</td>
<td>1.8</td>
<td>62.2</td>
<td>60</td>
<td>6.6</td>
<td>218.4</td>
<td><span class="tooltip" title="292.8 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.04</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">55</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.53</td>
<td>0.47</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>6</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Shadow_Demon" title="Shadow Demon"><img alt="Shadow Demon minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/35/Shadow_Demon_minimap_icon.png/20px-Shadow_Demon_minimap_icon.png?version=51e5d82481d2bd816a0f052f86b6be76" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/35/Shadow_Demon_minimap_icon.png/30px-Shadow_Demon_minimap_icon.png?version=51e5d82481d2bd816a0f052f86b6be76 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/3/35/Shadow_Demon_minimap_icon.png?version=51e5d82481d2bd816a0f052f86b6be76 2x"></a> <a href="/Shadow_Demon" title="Shadow Demon">Shadow Demon</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>23</td>
<td>2.6</td>
<td>85.4</td>
<td>18</td>
<td>2.2</td>
<td>70.8</td>
<td>21</td>
<td>3.3</td>
<td>100.2</td>
<td>62</td>
<td>8.1</td>
<td>256.4</td>
<td><span class="tooltip" title="292.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.88</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Shadow_Fiend" title="Shadow Fiend"><img alt="Shadow Fiend minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/00/Shadow_Fiend_minimap_icon.png/20px-Shadow_Fiend_minimap_icon.png?version=ce39474b8c3289b362a247001b09673a" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/00/Shadow_Fiend_minimap_icon.png/30px-Shadow_Fiend_minimap_icon.png?version=ce39474b8c3289b362a247001b09673a 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/00/Shadow_Fiend_minimap_icon.png?version=ce39474b8c3289b362a247001b09673a 2x"></a> <a href="/Shadow_Fiend" title="Shadow Fiend">Shadow Fiend</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>19</td>
<td>2.7</td>
<td>83.8</td>
<td>20</td>
<td>3.5</td>
<td>104</td>
<td>18</td>
<td>2.2</td>
<td>70.8</td>
<td>57</td>
<td>8.4</td>
<td>258.6</td>
<td><span class="tooltip" title="308.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.2</span></td>
<td><span class="tooltip" title="15 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">35</span></td>
<td><span class="tooltip" title="21 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">41</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.54</td>
<td>1800</td>
<td>800</td>
<td>1</td>
<td>24</td>
<td>0.25</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Shadow_Shaman" title="Shadow Shaman"><img alt="Shadow Shaman minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7e/Shadow_Shaman_minimap_icon.png/20px-Shadow_Shaman_minimap_icon.png?version=9c3b8e014b1fb4734a9d148add5545cd" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7e/Shadow_Shaman_minimap_icon.png/30px-Shadow_Shaman_minimap_icon.png?version=9c3b8e014b1fb4734a9d148add5545cd 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/7e/Shadow_Shaman_minimap_icon.png?version=9c3b8e014b1fb4734a9d148add5545cd 2x"></a> <a href="/Shadow_Shaman" title="Shadow Shaman">Shadow Shaman</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>23</td>
<td>2.3</td>
<td>78.2</td>
<td>16</td>
<td>1.6</td>
<td>54.4</td>
<td>19</td>
<td>3.5</td>
<td>103</td>
<td>58</td>
<td>7.4</td>
<td>235.6</td>
<td><span class="tooltip" title="287.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.56</span></td>
<td><span class="tooltip" title="52 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">71</span></td>
<td><span class="tooltip" title="59 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">78</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">400</span></td>
<td>90</td>
<td>1.7</td>
<td>0.3</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Silencer" title="Silencer"><img alt="Silencer minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0f/Silencer_minimap_icon.png/20px-Silencer_minimap_icon.png?version=17170f0633dfac3988e2a126500fb40b" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0f/Silencer_minimap_icon.png/30px-Silencer_minimap_icon.png?version=17170f0633dfac3988e2a126500fb40b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/0f/Silencer_minimap_icon.png?version=17170f0633dfac3988e2a126500fb40b 2x"></a> <a href="/Silencer" title="Silencer">Silencer</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>19</td>
<td>2.7</td>
<td>83.8</td>
<td>22</td>
<td>2.4</td>
<td>79.6</td>
<td>25</td>
<td>2.9</td>
<td>94.6</td>
<td>66</td>
<td>8</td>
<td>258</td>
<td><span class="tooltip" title="293.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.52</span></td>
<td><span class="tooltip" title="18 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">43</span></td>
<td><span class="tooltip" title="32 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>115</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Skywrath_Mage" title="Skywrath Mage"><img alt="Skywrath Mage minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/84/Skywrath_Mage_minimap_icon.png/20px-Skywrath_Mage_minimap_icon.png?version=72e6ce9c824eac8962c0a35411228d67" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/84/Skywrath_Mage_minimap_icon.png/30px-Skywrath_Mage_minimap_icon.png?version=72e6ce9c824eac8962c0a35411228d67 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/8/84/Skywrath_Mage_minimap_icon.png?version=72e6ce9c824eac8962c0a35411228d67 2x"></a> <a href="/Skywrath_Mage" title="Skywrath Mage">Skywrath Mage</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2</td>
<td>69</td>
<td>13</td>
<td>0.8</td>
<td>32.2</td>
<td>25</td>
<td>4.1</td>
<td>123.4</td>
<td>59</td>
<td>6.9</td>
<td>224.6</td>
<td><span class="tooltip" title="327.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">325</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0.08</span></td>
<td><span class="tooltip" title="14 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">39</span></td>
<td><span class="tooltip" title="24 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="1000 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">625</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.78</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Slardar" title="Slardar"><img alt="Slardar minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/ef/Slardar_minimap_icon.png/20px-Slardar_minimap_icon.png?version=56a40db7ff30ca25a0962529ec6f1198" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/ef/Slardar_minimap_icon.png/30px-Slardar_minimap_icon.png?version=56a40db7ff30ca25a0962529ec6f1198 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/e/ef/Slardar_minimap_icon.png?version=56a40db7ff30ca25a0962529ec6f1198 2x"></a> <a href="/Slardar" title="Slardar">Slardar</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>21</td>
<td>3.6</td>
<td>107.4</td>
<td>17</td>
<td>2.4</td>
<td>74.6</td>
<td>15</td>
<td>1.5</td>
<td>51</td>
<td>53</td>
<td>7.5</td>
<td>233</td>
<td><span class="tooltip" title="292.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.72</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="38 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.36</td>
<td>0.64</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.5</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Slark" title="Slark"><img alt="Slark minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1c/Slark_minimap_icon.png/20px-Slark_minimap_icon.png?version=acca6d391ee79e1c8c7a34c438924cb5" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1c/Slark_minimap_icon.png/30px-Slark_minimap_icon.png?version=acca6d391ee79e1c8c7a34c438924cb5 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/1c/Slark_minimap_icon.png?version=acca6d391ee79e1c8c7a34c438924cb5 2x"></a> <a href="/Slark" title="Slark">Slark</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>21</td>
<td>1.9</td>
<td>66.6</td>
<td>21</td>
<td>1.7</td>
<td>61.8</td>
<td>16</td>
<td>1.7</td>
<td>56.8</td>
<td>58</td>
<td>5.3</td>
<td>185.2</td>
<td><span class="tooltip" title="303.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.36</span></td>
<td><span class="tooltip" title="32 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="40 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>120</td>
<td>1.7</td>
<td>0.5</td>
<td>0.3</td>
<td>1800</td>
<td>1800</td>
<td>0.7</td>
<td>24</td>
<td>1.75</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Snapfire" title="Snapfire"><img alt="Snapfire minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/e1/Snapfire_minimap_icon.png/20px-Snapfire_minimap_icon.png?version=3064537447ac6594e91ae6dc550394d1" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/e1/Snapfire_minimap_icon.png/30px-Snapfire_minimap_icon.png?version=3064537447ac6594e91ae6dc550394d1 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/e/e1/Snapfire_minimap_icon.png?version=3064537447ac6594e91ae6dc550394d1 2x"></a> <a href="/Snapfire" title="Snapfire">Snapfire</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>20</td>
<td>3.3</td>
<td>99.2</td>
<td>16</td>
<td>1.9</td>
<td>61.6</td>
<td>18</td>
<td>2.2</td>
<td>70.8</td>
<td>54</td>
<td>7.4</td>
<td>231.6</td>
<td><span class="tooltip" title="302.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.56</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="1800 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.6</td>
<td>1</td>
<td>1</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Sniper" title="Sniper"><img alt="Sniper minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/f1/Sniper_minimap_icon.png/20px-Sniper_minimap_icon.png?version=23f29f0c39ecd59cead1e3a14996cd5d" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/f/f1/Sniper_minimap_icon.png/30px-Sniper_minimap_icon.png?version=23f29f0c39ecd59cead1e3a14996cd5d 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/f/f1/Sniper_minimap_icon.png?version=23f29f0c39ecd59cead1e3a14996cd5d 2x"></a> <a href="/Sniper" title="Sniper">Sniper</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>19</td>
<td>1.7</td>
<td>59.8</td>
<td>21</td>
<td>3.4</td>
<td>102.6</td>
<td>15</td>
<td>2.6</td>
<td>77.4</td>
<td>55</td>
<td>7.7</td>
<td>239.8</td>
<td><span class="tooltip" title="288 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.36</span></td>
<td><span class="tooltip" title="15 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">36</span></td>
<td><span class="tooltip" title="21 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="3000 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">550</span></td>
<td>100</td>
<td>1.7</td>
<td>0.17</td>
<td>0.7</td>
<td>1800</td>
<td>1400</td>
<td>0.7</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Spectre" title="Spectre"><img alt="Spectre minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/64/Spectre_minimap_icon.png/20px-Spectre_minimap_icon.png?version=9af3fa85c3d28eb3b856e10d4e2e5717" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/64/Spectre_minimap_icon.png/30px-Spectre_minimap_icon.png?version=9af3fa85c3d28eb3b856e10d4e2e5717 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/64/Spectre_minimap_icon.png?version=9af3fa85c3d28eb3b856e10d4e2e5717 2x"></a> <a href="/Spectre" title="Spectre">Spectre</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>23</td>
<td>2.5</td>
<td>83</td>
<td>23</td>
<td>2.1</td>
<td>73.4</td>
<td>16</td>
<td>1.9</td>
<td>61.6</td>
<td>62</td>
<td>6.5</td>
<td>218</td>
<td><span class="tooltip" title="293.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.68</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="27 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Spirit_Breaker" title="Spirit Breaker"><img alt="Spirit Breaker minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/90/Spirit_Breaker_minimap_icon.png/20px-Spirit_Breaker_minimap_icon.png?version=4ea7f0484fde99d08d9026ff0fe19bf0" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/90/Spirit_Breaker_minimap_icon.png/30px-Spirit_Breaker_minimap_icon.png?version=4ea7f0484fde99d08d9026ff0fe19bf0 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/90/Spirit_Breaker_minimap_icon.png?version=4ea7f0484fde99d08d9026ff0fe19bf0 2x"></a> <a href="/Spirit_Breaker" title="Spirit Breaker">Spirit Breaker</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>29</td>
<td>3.1</td>
<td>103.4</td>
<td>17</td>
<td>1.7</td>
<td>57.8</td>
<td>14</td>
<td>1.8</td>
<td>57.2</td>
<td>60</td>
<td>6.6</td>
<td>218.4</td>
<td><span class="tooltip" title="287.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.72</span></td>
<td><span class="tooltip" title="31 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">70</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.9</td>
<td>0.6</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>1.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Storm_Spirit" title="Storm Spirit"><img alt="Storm Spirit minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d6/Storm_Spirit_minimap_icon.png/20px-Storm_Spirit_minimap_icon.png?version=16efe99893a055a146de621443bfd855" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d6/Storm_Spirit_minimap_icon.png/30px-Storm_Spirit_minimap_icon.png?version=16efe99893a055a146de621443bfd855 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/d/d6/Storm_Spirit_minimap_icon.png?version=16efe99893a055a146de621443bfd855 2x"></a> <a href="/Storm_Spirit" title="Storm Spirit">Storm Spirit</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2</td>
<td>69</td>
<td>22</td>
<td>1.5</td>
<td>58</td>
<td>23</td>
<td>3.9</td>
<td>116.6</td>
<td>66</td>
<td>7.4</td>
<td>243.6</td>
<td><span class="tooltip" title="288.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.52</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="1100 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">480</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.8</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Sven" title="Sven"><img alt="Sven minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1c/Sven_minimap_icon.png/20px-Sven_minimap_icon.png?version=99afd55f19f7a188e414a1105428116f" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1c/Sven_minimap_icon.png/30px-Sven_minimap_icon.png?version=99afd55f19f7a188e414a1105428116f 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/1c/Sven_minimap_icon.png?version=99afd55f19f7a188e414a1105428116f 2x"></a> <a href="/Sven" title="Sven">Sven</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>3.2</td>
<td>98.8</td>
<td>21</td>
<td>2</td>
<td>69</td>
<td>16</td>
<td>1.3</td>
<td>47.2</td>
<td>59</td>
<td>6.5</td>
<td>215</td>
<td><span class="tooltip" title="313.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.36</span></td>
<td><span class="tooltip" title="41 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="43 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">65</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.8</td>
<td>0.4</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Techies" title="Techies"><img alt="Techies minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/0c/Techies_minimap_icon.png/20px-Techies_minimap_icon.png?version=f5931b673cb85c688ef0d70c41dba322" width="20" height="21" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/0/0c/Techies_minimap_icon.png?version=f5931b673cb85c688ef0d70c41dba322 1.5x"></a> <a href="/Techies" title="Techies">Techies</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>19</td>
<td>2.5</td>
<td>79</td>
<td>14</td>
<td>1.3</td>
<td>45.2</td>
<td>25</td>
<td>3.3</td>
<td>104.2</td>
<td>58</td>
<td>7.1</td>
<td>228.4</td>
<td><span class="tooltip" title="312.2 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="5 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">7.24</span></td>
<td><span class="tooltip" title="9 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">34</span></td>
<td><span class="tooltip" title="11 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">36</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">700</span></td>
<td>100</td>
<td>1.7</td>
<td>0.5</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>6</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Templar_Assassin" title="Templar Assassin"><img alt="Templar Assassin minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/6b/Templar_Assassin_minimap_icon.png/20px-Templar_Assassin_minimap_icon.png?version=8f30f4bf05851f61fbdb2c2e36197e9a" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/6/6b/Templar_Assassin_minimap_icon.png/30px-Templar_Assassin_minimap_icon.png?version=8f30f4bf05851f61fbdb2c2e36197e9a 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/6/6b/Templar_Assassin_minimap_icon.png?version=8f30f4bf05851f61fbdb2c2e36197e9a 2x"></a> <a href="/Templar_Assassin" title="Templar Assassin">Templar Assassin</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>19</td>
<td>2.4</td>
<td>76.6</td>
<td>23</td>
<td>3.2</td>
<td>99.8</td>
<td>20</td>
<td>2</td>
<td>68</td>
<td>62</td>
<td>7.6</td>
<td>244.4</td>
<td><span class="tooltip" title="308.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.68</span></td>
<td><span class="tooltip" title="30 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">140</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Terrorblade" title="Terrorblade"><img alt="Terrorblade minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/72/Terrorblade_minimap_icon.png/20px-Terrorblade_minimap_icon.png?version=85d5861a07f4048d4a9194643ae9ec21" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/72/Terrorblade_minimap_icon.png/30px-Terrorblade_minimap_icon.png?version=85d5861a07f4048d4a9194643ae9ec21 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/7/72/Terrorblade_minimap_icon.png?version=85d5861a07f4048d4a9194643ae9ec21 2x"></a> <a href="/Terrorblade" title="Terrorblade">Terrorblade</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>15</td>
<td>1.7</td>
<td>55.8</td>
<td>22</td>
<td>4.8</td>
<td>137.2</td>
<td>19</td>
<td>1.6</td>
<td>57.4</td>
<td>56</td>
<td>8.1</td>
<td>250.4</td>
<td><span class="tooltip" title="313.4 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="7 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">10.52</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">48</span></td>
<td><span class="tooltip" title="32 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.5</td>
<td>0.3</td>
<td>0.6</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Tidehunter" title="Tidehunter"><img alt="Tidehunter minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d4/Tidehunter_minimap_icon.png/20px-Tidehunter_minimap_icon.png?version=e3fe220ff123bcbd7ad921679d542731" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d4/Tidehunter_minimap_icon.png/30px-Tidehunter_minimap_icon.png?version=e3fe220ff123bcbd7ad921679d542731 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/d/d4/Tidehunter_minimap_icon.png?version=e3fe220ff123bcbd7ad921679d542731 2x"></a> <a href="/Tidehunter" title="Tidehunter">Tidehunter</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>3.5</td>
<td>106</td>
<td>15</td>
<td>1.5</td>
<td>51</td>
<td>16</td>
<td>1.7</td>
<td>56.8</td>
<td>53</td>
<td>6.7</td>
<td>213.8</td>
<td><span class="tooltip" title="302.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.4</span></td>
<td><span class="tooltip" title="25 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">47</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.6</td>
<td>0.56</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Timbersaw" title="Timbersaw"><img alt="Timbersaw minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/49/Timbersaw_minimap_icon.png/20px-Timbersaw_minimap_icon.png?version=8f03cc2d8b0ec82d636b62a41cb53df7" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/4/49/Timbersaw_minimap_icon.png/30px-Timbersaw_minimap_icon.png?version=8f03cc2d8b0ec82d636b62a41cb53df7 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/4/49/Timbersaw_minimap_icon.png?version=8f03cc2d8b0ec82d636b62a41cb53df7 2x"></a> <a href="/Timbersaw" title="Timbersaw">Timbersaw</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>3.2</td>
<td>99.8</td>
<td>16</td>
<td>1.6</td>
<td>54.4</td>
<td>23</td>
<td>2.7</td>
<td>87.8</td>
<td>62</td>
<td>7.5</td>
<td>242</td>
<td><span class="tooltip" title="292.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.56</span></td>
<td><span class="tooltip" title="26 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="30 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">53</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.36</td>
<td>0.64</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Tinker" title="Tinker"><img alt="Tinker minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1b/Tinker_minimap_icon.png/20px-Tinker_minimap_icon.png?version=98579483ad3a6d45dc3d79a11a0d5264" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/1/1b/Tinker_minimap_icon.png/30px-Tinker_minimap_icon.png?version=98579483ad3a6d45dc3d79a11a0d5264 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/1/1b/Tinker_minimap_icon.png?version=98579483ad3a6d45dc3d79a11a0d5264 2x"></a> <a href="/Tinker" title="Tinker">Tinker</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.5</td>
<td>78</td>
<td>13</td>
<td>1.2</td>
<td>41.8</td>
<td>30</td>
<td>3.3</td>
<td>109.2</td>
<td>61</td>
<td>7</td>
<td>229</td>
<td><span class="tooltip" title="291.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.08</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="30 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>0.65</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Tiny" title="Tiny"><img alt="Tiny minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b2/Tiny_minimap_icon.png/20px-Tiny_minimap_icon.png?version=2add520985b5dee97795671e1e60f864" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b2/Tiny_minimap_icon.png/30px-Tiny_minimap_icon.png?version=2add520985b5dee97795671e1e60f864 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/b2/Tiny_minimap_icon.png?version=2add520985b5dee97795671e1e60f864 2x"></a> <a href="/Tiny" title="Tiny">Tiny</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>28</td>
<td>4.1</td>
<td>126.4</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>17</td>
<td>2.2</td>
<td>69.8</td>
<td>45</td>
<td>6.3</td>
<td>196.2</td>
<td><span class="tooltip" title="310 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">0</span></td>
<td><span class="tooltip" title="46 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">74</span></td>
<td><span class="tooltip" title="52 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">80</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>1</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Treant_Protector" title="Treant Protector"><img alt="Treant Protector minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8f/Treant_Protector_minimap_icon.png/20px-Treant_Protector_minimap_icon.png?version=1808a1dc4371ffcc6ac8bd0be3ad837f" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8f/Treant_Protector_minimap_icon.png/30px-Treant_Protector_minimap_icon.png?version=1808a1dc4371ffcc6ac8bd0be3ad837f 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/8/8f/Treant_Protector_minimap_icon.png?version=1808a1dc4371ffcc6ac8bd0be3ad837f 2x"></a> <a href="/Treant_Protector" title="Treant Protector">Treant Protector</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>25</td>
<td>4.1</td>
<td>123.4</td>
<td>15</td>
<td>2</td>
<td>63</td>
<td>20</td>
<td>1.8</td>
<td>63.2</td>
<td>60</td>
<td>7.9</td>
<td>249.6</td>
<td><span class="tooltip" title="282.1 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">280</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.4</span></td>
<td><span class="tooltip" title="62 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">87</span></td>
<td><span class="tooltip" title="70 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">95</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.9</td>
<td>0.6</td>
<td>0.4</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Troll_Warlord" title="Troll Warlord"><img alt="Troll Warlord minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a2/Troll_Warlord_minimap_icon.png/20px-Troll_Warlord_minimap_icon.png?version=cb0f673063283426d1808ec25eaacbe3" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/a/a2/Troll_Warlord_minimap_icon.png/30px-Troll_Warlord_minimap_icon.png?version=cb0f673063283426d1808ec25eaacbe3 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/a/a2/Troll_Warlord_minimap_icon.png?version=cb0f673063283426d1808ec25eaacbe3 2x"></a> <a href="/Troll_Warlord" title="Troll Warlord">Troll Warlord</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>21</td>
<td>2.5</td>
<td>81</td>
<td>21</td>
<td>3.3</td>
<td>100.2</td>
<td>13</td>
<td>1</td>
<td>37</td>
<td>55</td>
<td>6.8</td>
<td>218.2</td>
<td><span class="tooltip" title="293 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.36</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="35 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">500</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Tusk" title="Tusk"><img alt="Tusk minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/c3/Tusk_minimap_icon.png/20px-Tusk_minimap_icon.png?version=8256a428049384053d45134ff4964667" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/c3/Tusk_minimap_icon.png/30px-Tusk_minimap_icon.png?version=8256a428049384053d45134ff4964667 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/c/c3/Tusk_minimap_icon.png?version=8256a428049384053d45134ff4964667 2x"></a> <a href="/Tusk" title="Tusk">Tusk</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>23</td>
<td>3.9</td>
<td>116.6</td>
<td>23</td>
<td>2.1</td>
<td>73.4</td>
<td>18</td>
<td>1.7</td>
<td>58.8</td>
<td>64</td>
<td>7.7</td>
<td>248.8</td>
<td><span class="tooltip" title="313.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">310</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.68</span></td>
<td><span class="tooltip" title="27 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="31 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.36</td>
<td>0.64</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Underlord" title="Underlord"><img alt="Underlord minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/29/Underlord_minimap_icon.png/20px-Underlord_minimap_icon.png?version=a7f8280a97002261f90380ceb0b1864d" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/29/Underlord_minimap_icon.png/30px-Underlord_minimap_icon.png?version=a7f8280a97002261f90380ceb0b1864d 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/29/Underlord_minimap_icon.png?version=a7f8280a97002261f90380ceb0b1864d 2x"></a> <a href="/Underlord" title="Underlord">Underlord</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>25</td>
<td>3.3</td>
<td>104.2</td>
<td>12</td>
<td>1.6</td>
<td>50.4</td>
<td>17</td>
<td>2.3</td>
<td>72.2</td>
<td>54</td>
<td>7.2</td>
<td>226.8</td>
<td><span class="tooltip" title="296.8 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="3 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.92</span></td>
<td><span class="tooltip" title="37 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">62</span></td>
<td><span class="tooltip" title="43 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">68</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.45</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Undying" title="Undying"><img alt="Undying minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8a/Undying_minimap_icon.png/20px-Undying_minimap_icon.png?version=0cd4df25ff2047eead225bf0811deb74" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/8/8a/Undying_minimap_icon.png/30px-Undying_minimap_icon.png?version=0cd4df25ff2047eead225bf0811deb74 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/8/8a/Undying_minimap_icon.png?version=0cd4df25ff2047eead225bf0811deb74 2x"></a> <a href="/Undying" title="Undying">Undying</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>2.8</td>
<td>89.2</td>
<td>10</td>
<td>0.8</td>
<td>29.2</td>
<td>27</td>
<td>2.8</td>
<td>94.2</td>
<td>59</td>
<td>6.4</td>
<td>212.6</td>
<td><span class="tooltip" title="301.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.6</span></td>
<td><span class="tooltip" title="35 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">57</span></td>
<td><span class="tooltip" title="43 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">65</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Ursa" title="Ursa"><img alt="Ursa minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/ee/Ursa_minimap_icon.png/20px-Ursa_minimap_icon.png?version=e7e811986e976d7984c0d9933252a0b1" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/e/ee/Ursa_minimap_icon.png/30px-Ursa_minimap_icon.png?version=e7e811986e976d7984c0d9933252a0b1 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/e/ee/Ursa_minimap_icon.png?version=e7e811986e976d7984c0d9933252a0b1 2x"></a> <a href="/Ursa" title="Ursa">Ursa</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>24</td>
<td>3.1</td>
<td>98.4</td>
<td>18</td>
<td>2.4</td>
<td>75.6</td>
<td>16</td>
<td>1.5</td>
<td>52</td>
<td>58</td>
<td>7</td>
<td>226</td>
<td><span class="tooltip" title="317.8 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">315</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.88</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="28 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0.5</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Vengeful_Spirit" title="Vengeful Spirit"><img alt="Vengeful Spirit minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/09/Vengeful_Spirit_minimap_icon.png/20px-Vengeful_Spirit_minimap_icon.png?version=98520cdeb384d0e9e9fd809e1b892a94" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/09/Vengeful_Spirit_minimap_icon.png/30px-Vengeful_Spirit_minimap_icon.png?version=98520cdeb384d0e9e9fd809e1b892a94 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/09/Vengeful_Spirit_minimap_icon.png?version=98520cdeb384d0e9e9fd809e1b892a94 2x"></a> <a href="/Vengeful_Spirit" title="Vengeful Spirit">Vengeful Spirit</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>19</td>
<td>2.6</td>
<td>81.4</td>
<td>20</td>
<td>3.8</td>
<td>111.2</td>
<td>17</td>
<td>1.5</td>
<td>53</td>
<td>56</td>
<td>7.9</td>
<td>245.6</td>
<td><span class="tooltip" title="292.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">3.2</span></td>
<td><span class="tooltip" title="14 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">34</span></td>
<td><span class="tooltip" title="22 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="1500 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">400</span></td>
<td>100</td>
<td>1.7</td>
<td>0.33</td>
<td>0.64</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0.25</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Venomancer" title="Venomancer"><img alt="Venomancer minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/95/Venomancer_minimap_icon.png/20px-Venomancer_minimap_icon.png?version=691cdf5d4ff048b8437f32be37e700ee" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/9/95/Venomancer_minimap_icon.png/30px-Venomancer_minimap_icon.png?version=691cdf5d4ff048b8437f32be37e700ee 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/9/95/Venomancer_minimap_icon.png?version=691cdf5d4ff048b8437f32be37e700ee 2x"></a> <a href="/Venomancer" title="Venomancer">Venomancer</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>19</td>
<td>2.1</td>
<td>69.4</td>
<td>26</td>
<td>3.2</td>
<td>102.8</td>
<td>19</td>
<td>1.8</td>
<td>62.2</td>
<td>64</td>
<td>7.1</td>
<td>234.4</td>
<td><span class="tooltip" title="278.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">275</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">4.16</span></td>
<td><span class="tooltip" title="16 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="18 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">44</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">450</span></td>
<td>115</td>
<td>1.7</td>
<td>0.3</td>
<td>0.7</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Viper" title="Viper"><img alt="Viper minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/cb/Viper_minimap_icon.png/20px-Viper_minimap_icon.png?version=d10dabfe69ffcba2bb334e2f3a40b631" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/c/cb/Viper_minimap_icon.png/30px-Viper_minimap_icon.png?version=d10dabfe69ffcba2bb334e2f3a40b631 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/c/cb/Viper_minimap_icon.png?version=d10dabfe69ffcba2bb334e2f3a40b631 2x"></a> <a href="/Viper" title="Viper">Viper</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>21</td>
<td>2.4</td>
<td>78.6</td>
<td>21</td>
<td>2.5</td>
<td>81</td>
<td>15</td>
<td>1.8</td>
<td>58.2</td>
<td>57</td>
<td>6.7</td>
<td>217.8</td>
<td><span class="tooltip" title="277.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">275</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.36</span></td>
<td><span class="tooltip" title="25 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">46</span></td>
<td><span class="tooltip" title="27 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">48</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">575</span></td>
<td>120</td>
<td>1.7</td>
<td>0.33</td>
<td>1</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>0</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Visage" title="Visage"><img alt="Visage minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2f/Visage_minimap_icon.png/20px-Visage_minimap_icon.png?version=65e9e0c4a2db29e6a5d7dc484eb7ac05" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2f/Visage_minimap_icon.png/30px-Visage_minimap_icon.png?version=65e9e0c4a2db29e6a5d7dc484eb7ac05 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/2f/Visage_minimap_icon.png?version=65e9e0c4a2db29e6a5d7dc484eb7ac05 2x"></a> <a href="/Visage" title="Visage">Visage</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>22</td>
<td>2.8</td>
<td>89.2</td>
<td>11</td>
<td>1.3</td>
<td>42.2</td>
<td>22</td>
<td>2.9</td>
<td>91.6</td>
<td>55</td>
<td>7</td>
<td>223</td>
<td><span class="tooltip" title="286.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="-2 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">-0.24</span></td>
<td><span class="tooltip" title="23 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="33 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">55</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.46</td>
<td>0.54</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Void_Spirit" title="Void Spirit"><img alt="Void Spirit minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/02/Void_Spirit_minimap_icon.png/20px-Void_Spirit_minimap_icon.png?version=ca73afd9a36e8d492ce2c352d32fb3ac" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/02/Void_Spirit_minimap_icon.png/30px-Void_Spirit_minimap_icon.png?version=ca73afd9a36e8d492ce2c352d32fb3ac 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/02/Void_Spirit_minimap_icon.png?version=ca73afd9a36e8d492ce2c352d32fb3ac 2x"></a> <a href="/Void_Spirit" title="Void Spirit">Void Spirit</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>22</td>
<td>2.8</td>
<td>89.2</td>
<td>19</td>
<td>2.2</td>
<td>71.8</td>
<td>24</td>
<td>3.1</td>
<td>98.4</td>
<td>65</td>
<td>8.1</td>
<td>259.4</td>
<td><span class="tooltip" title="307.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">305</span></td>
<td><span class="tooltip" title="2.66 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">5.7</span></td>
<td><span class="tooltip" title="32 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">56</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">200</span></td>
<td>100</td>
<td>1.7</td>
<td>0.35</td>
<td>1</td>
<td>1800</td>
<td>800</td>
<td>0.7</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Warlock" title="Warlock"><img alt="Warlock minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d3/Warlock_minimap_icon.png/20px-Warlock_minimap_icon.png?version=c927f107086e7e06ecec122e1fbe90d8" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d3/Warlock_minimap_icon.png/30px-Warlock_minimap_icon.png?version=c927f107086e7e06ecec122e1fbe90d8 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/d/d3/Warlock_minimap_icon.png?version=c927f107086e7e06ecec122e1fbe90d8 2x"></a> <a href="/Warlock" title="Warlock">Warlock</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>26</td>
<td>3</td>
<td>98</td>
<td>10</td>
<td>1</td>
<td>34</td>
<td>25</td>
<td>3.1</td>
<td>99.4</td>
<td>61</td>
<td>7.1</td>
<td>231.4</td>
<td><span class="tooltip" title="291.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.6</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">49</span></td>
<td><span class="tooltip" title="34 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">59</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.3</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Weaver" title="Weaver"><img alt="Weaver minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d5/Weaver_minimap_icon.png/20px-Weaver_minimap_icon.png?version=1253d1716cf3e488e6b0980d7e919298" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/d5/Weaver_minimap_icon.png/30px-Weaver_minimap_icon.png?version=1253d1716cf3e488e6b0980d7e919298 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/d/d5/Weaver_minimap_icon.png?version=1253d1716cf3e488e6b0980d7e919298 2x"></a> <a href="/Weaver" title="Weaver">Weaver</a></span></td>
<td><a href="/Agility" title="Agility"><img alt="Agility attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/20px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/30px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/2d/Agility_attribute_symbol.png/40px-Agility_attribute_symbol.png?version=4022c3f8fd6aec761efd859358f8c078 2x"></a></td>
<td>16</td>
<td>2</td>
<td>64</td>
<td>14</td>
<td>3.6</td>
<td>100.4</td>
<td>13</td>
<td>1.8</td>
<td>56.2</td>
<td>43</td>
<td>7.4</td>
<td>220.6</td>
<td><span class="tooltip" title="276.9 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">275</span></td>
<td><span class="tooltip" title="0 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.24</span></td>
<td><span class="tooltip" title="36 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">50</span></td>
<td><span class="tooltip" title="46 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">60</span></td>
<td><span class="tooltip" title="900 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">425</span></td>
<td>120</td>
<td>1.8</td>
<td>0.64</td>
<td>0.36</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>1</td>
<td>4</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Windranger" title="Windranger"><img alt="Windranger minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/df/Windranger_minimap_icon.png/20px-Windranger_minimap_icon.png?version=dc6694c8fcdd4a36014ac565273089ba" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/d/df/Windranger_minimap_icon.png/30px-Windranger_minimap_icon.png?version=dc6694c8fcdd4a36014ac565273089ba 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/d/df/Windranger_minimap_icon.png?version=dc6694c8fcdd4a36014ac565273089ba 2x"></a> <a href="/Windranger" title="Windranger">Windranger</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>3</td>
<td>90</td>
<td>17</td>
<td>1.4</td>
<td>50.6</td>
<td>18</td>
<td>3</td>
<td>90</td>
<td>53</td>
<td>7.4</td>
<td>230.6</td>
<td><span class="tooltip" title="292.5 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">290</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.72</span></td>
<td><span class="tooltip" title="24 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">42</span></td>
<td><span class="tooltip" title="36 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">54</span></td>
<td><span class="tooltip" title="1250 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.5</td>
<td>0.4</td>
<td>0.3</td>
<td>1800</td>
<td>800</td>
<td>0.8</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Winter_Wyvern" title="Winter Wyvern"><img alt="Winter Wyvern minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/06/Winter_Wyvern_minimap_icon.png/20px-Winter_Wyvern_minimap_icon.png?version=9acb8ff16a91cd7604625e62de6c460d" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/0/06/Winter_Wyvern_minimap_icon.png/30px-Winter_Wyvern_minimap_icon.png?version=9acb8ff16a91cd7604625e62de6c460d 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/0/06/Winter_Wyvern_minimap_icon.png?version=9acb8ff16a91cd7604625e62de6c460d 2x"></a> <a href="/Winter_Wyvern" title="Winter Wyvern">Winter Wyvern</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>26</td>
<td>2.6</td>
<td>88.4</td>
<td>16</td>
<td>1.9</td>
<td>61.6</td>
<td>26</td>
<td>3.6</td>
<td>112.4</td>
<td>68</td>
<td>8.1</td>
<td>262.4</td>
<td><span class="tooltip" title="287.3 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">285</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.56</span></td>
<td><span class="tooltip" title="12 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">38</span></td>
<td><span class="tooltip" title="19 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">45</span></td>
<td><span class="tooltip" title="700 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">425</span></td>
<td>100</td>
<td>1.7</td>
<td>0.25</td>
<td>0.8</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Witch_Doctor" title="Witch Doctor"><img alt="Witch Doctor minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b6/Witch_Doctor_minimap_icon.png/20px-Witch_Doctor_minimap_icon.png?version=dcdab252e35d0465de963a44cc2120c6" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/b/b6/Witch_Doctor_minimap_icon.png/30px-Witch_Doctor_minimap_icon.png?version=dcdab252e35d0465de963a44cc2120c6 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/b/b6/Witch_Doctor_minimap_icon.png?version=dcdab252e35d0465de963a44cc2120c6 2x"></a> <a href="/Witch_Doctor" title="Witch Doctor">Witch Doctor</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>18</td>
<td>2.3</td>
<td>73.2</td>
<td>13</td>
<td>1.4</td>
<td>46.6</td>
<td>22</td>
<td>3.3</td>
<td>101.2</td>
<td>53</td>
<td>7</td>
<td>221</td>
<td><span class="tooltip" title="302 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">300</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.08</span></td>
<td><span class="tooltip" title="29 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">51</span></td>
<td><span class="tooltip" title="39 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="1200 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">600</span></td>
<td>100</td>
<td>1.7</td>
<td>0.4</td>
<td>0.5</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Wraith_King" title="Wraith King"><img alt="Wraith King minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/22/Wraith_King_minimap_icon.png/20px-Wraith_King_minimap_icon.png?version=0e605207bc39ff3fcb84cf50667818b9" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/2/22/Wraith_King_minimap_icon.png/30px-Wraith_King_minimap_icon.png?version=0e605207bc39ff3fcb84cf50667818b9 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/2/22/Wraith_King_minimap_icon.png?version=0e605207bc39ff3fcb84cf50667818b9 2x"></a> <a href="/Wraith_King" title="Wraith King">Wraith King</a></span></td>
<td><a href="/Strength" title="Strength"><img alt="Strength attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/20px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/30px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/7/7a/Strength_attribute_symbol.png/40px-Strength_attribute_symbol.png?version=ffbadd5d5525cbe8e2cf8e4c24a2d115 2x"></a></td>
<td>22</td>
<td>3.2</td>
<td>98.8</td>
<td>18</td>
<td>1.7</td>
<td>58.8</td>
<td>18</td>
<td>1.6</td>
<td>56.4</td>
<td>58</td>
<td>6.5</td>
<td>214</td>
<td><span class="tooltip" title="317.8 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">315</span></td>
<td><span class="tooltip" title="-1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">1.88</span></td>
<td><span class="tooltip" title="39 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">61</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="Melee" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">150</span></td>
<td>100</td>
<td>1.7</td>
<td>0.56</td>
<td>0.44</td>
<td>1800</td>
<td>800</td>
<td>0.5</td>
<td>24</td>
<td>0</td>
<td>2</td></tr>
<tr>
<td style="text-align:left;"><span class="image-link"><a href="/Zeus" title="Zeus"><img alt="Zeus minimap icon.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/3a/Zeus_minimap_icon.png/20px-Zeus_minimap_icon.png?version=4b06b7c55cca103af9c30a3acc4f2343" width="20" height="20" class="noprint pixelart" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/3/3a/Zeus_minimap_icon.png/30px-Zeus_minimap_icon.png?version=4b06b7c55cca103af9c30a3acc4f2343 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/3/3a/Zeus_minimap_icon.png?version=4b06b7c55cca103af9c30a3acc4f2343 2x"></a> <a href="/Zeus" title="Zeus">Zeus</a></span></td>
<td><a href="/Intelligence" title="Intelligence"><img alt="Intelligence attribute symbol.png" src="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/20px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b" width="20" height="20" srcset="https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/30px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 1.5x, https://gamepedia.cursecdn.com/dota2_gamepedia/thumb/5/56/Intelligence_attribute_symbol.png/40px-Intelligence_attribute_symbol.png?version=794665f4224738e861793a93815a022b 2x"></a></td>
<td>21</td>
<td>2.1</td>
<td>71.4</td>
<td>11</td>
<td>1.2</td>
<td>39.8</td>
<td>22</td>
<td>3.3</td>
<td>101.2</td>
<td>54</td>
<td>6.6</td>
<td>212.4</td>
<td><span class="tooltip" title="296.6 with the bonus from base agility" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">295</span></td>
<td><span class="tooltip" title="1 Base armor" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">2.76</span></td>
<td><span class="tooltip" title="33 Minimum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">55</span></td>
<td><span class="tooltip" title="41 Maximum base attack damage" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">63</span></td>
<td><span class="tooltip" title="1100 Projectile speed" style="cursor: help; border-bottom: 1px dotted; --darkreader-inline-border-bottom: initial;" data-darkreader-inline-border-bottom="">380</span></td>
<td>100</td>
<td>1.7</td>
<td>0.45</td>
<td>0.55</td>
<td>1800</td>
<td>800</td>
<td>0.6</td>
<td>24</td>
<td>0</td>
<td>2</td></tr></tbody><tfoot></tfoot></table>
'''
parser.feed(tbl)
