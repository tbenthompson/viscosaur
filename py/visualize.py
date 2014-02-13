import time
import sys
sys.path.append("/home/tbent/packages/visit2_7_0/2.7.0/linux-x86_64/lib/site-packages/")
import visit
visit.Launch('/home/tbent/packages/visit2_7_0/bin/')
# for i in range(7):
#     time.sleep(1.5)
#     visit.DeleteAllPlots()
#     visit.OpenDatabase('./data/test/init_refinement_' + str(i) + '.0000.pvtu')
#     visit.DefineScalarExpression('logabstentszx', 'log(abs(tent_szx))')
#     # visit.AddPlot('Pseudocolor', 'logabstentszx')
#     visit.AddPlot('Pseudocolor', 'vel')
#     visit.AddPlot('Mesh', 'mesh')
#     visit.DrawPlots()


visit.DeleteAllPlots()
visit.OpenDatabase('./data/test/solution-4.0000.pvtu')
visit.DefineScalarExpression('logabstentszx', 'log(abs(tent_szx))')
visit.AddPlot('Pseudocolor', 'logabstentszx')
# visit.AddPlot('Pseudocolor', 'vel')
# visit.AddPlot('Mesh', 'mesh')
visit.DrawPlots()
