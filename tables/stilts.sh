#!/bin/bash

stilts tpipe \
  in=HECATEv2_DP.fits \
  omode=out out=HECATEv2_DP_extra.fits \
  cmd='addcol -units "log(Msol/yr)" -desc "Merge of log SFR" logSFR_total "logSFR_HECv2"'\
  cmd='addcol -units "Msol/yr" -desc "Total SFR in linear scale" SFR_total "exp10(logSFR_total)"'\
  cmd='addcol -units "log(Msol)" -desc "" logM_total "logM_star_HECv2"'\
  cmd='addcol -units "Msol" -desc "Total M_* in linear scale" M_total "exp10(logM_total)"'\
  cmd='addcol -units "1/yr" -desc "Specific SFR" sSFR "SFR_total/M_total"'\
  cmd='addcol -units "log10(1/yr)" -desc "Specific SFR in logarithmic scale" logsSFR "log10(sSFR)"'\
  cmd='addcol -desc "ID column" ID "Index"'\

stilts tpipe \
   in=HECATEv2_DP_extra.fits \
   out=Hec_2-10.csv \
   cmd='select "D_v2 > 2 && D_v2 < 11"'

stilts tpipe \
   in=HECATEv2_DP_extra.fits \
   out=Hec_10.csv \
   cmd='select "D_v2 < 11"'

stilts tpipe \
   in=HECATEv2_DP_extra.fits \
   out=Hec_11-20.csv \
   cmd='select "D_v2 >= 11  && D_v2 < 20"'

stilts tpipe \
   in=HECATEv2_DP_extra.fits \
   out=Hec_21-30.csv \
   cmd='select "D_v2 >= 21  && D_v2 < 30"'

stilts tpipe \
   in=HECATEv2_DP_extra.fits \
   out=Hec_31-40.csv \
   cmd='select "D_v2 >= 31  && D_v2 < 40"'

stilts tpipe \
   in=HECATEv2_DP_extra.fits \
   out=Hec_50.csv \
   cmd='select "D_v2 < 51"'

stilts tpipe \
   in=HECATEv2_DP_extra.fits \
   out=Hec_41-50.csv \
   cmd='select "D_v2 >= 41  && D_v2 < 51"'
