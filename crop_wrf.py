
from cdo import *
cdo = Cdo()
cdo.debug = True
out_file = 'wrfout'

date1 = '2019-11-29T00:00:00'
date2 = '2019-12-10T00:00:00'
# my target is in meters
z_levels = '0,350'
bbox_str = '-40.27287,-40.26817,-20.6817,-20.24395'

in_file = 'wrfout_d02_2019-11-29_00:00:00'
# crop time dimension to range:
cdo.seldate(
    f'{date1},{date2}', input=in_file, output=out_file, options='-f nc'
)

# crop to target bbox
cdo.sellonlatbox(bbox_str, input = out_file, output = out_file, options ='-f nc')

# crop z to an array of target levels:
cdo.sellevel(z_levels, input = in_file, output = out_file, options ='-f nc')

# chaining multiple operations:
cdo.seldate(
    f'{date1},{date2}',
    input=f' -sellonlatbox,{bbox_str} -sellevel,{z_levels} {in_file}',
    output=out_file,
    options='-f nc',
)
