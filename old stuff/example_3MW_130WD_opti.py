from spar_modified import Spar

example = Spar()
example.wall_thickness = [0.05, 0.05, 0.05, 0.05]
example.number_of_rings = [1., 4., 4., 13.]
example.neutral_axis = 0.2
example.water_depth = 130.
example.load_condition = 'N'
example.significant_wave_height = 10.660
example.significant_wave_period = 13.210
example.keel_cg_mooring = 51.
example.keel_cg_operating_system = 20.312
example.reference_wind_speed = 11.
example.reference_height = 75.
example.alpha = 0.110
example.material_density = 7850.
example.E = 200.e9
example.nu = 0.3
example.yield_stress = 345000000.
example.rotor_mass = 125000.
example.tower_mass = 83705.
example.free_board = 13.
example.draft = 64.
example.fixed_ballast_mass = 1244227.77
example.hull_mass = 890985.086
example.permanent_ballast_mass = 838450.256
example.variable_ballast_mass = 418535.462
example.number_of_sections = 4
example.outer_diameter = [5., 6., 6., 9.]
example.length = [6., 12., 15., 44.]
example.end_elevation = [7., -5., -20., -64.]
example.start_elevation = [13., 7., -5., -20.]
example.bulk_head = ['N', 'T', 'N', 'B']
example.execute()

class optimizationSpar(Assembly):
    def configure(self): 
        self.add('driver',SLSQPdriver())
        self.add('Spar',spar())
        self.driver.workflow.add('Spar')
        # objective
        self.driver.add_objective('Spar.SHM')
        # design variables
        self.driver.add_parameter('Spar.wall_thickness')
        self.driver.add_parameter('Spar.number_of_rings')
        self.driver.add_parameter('Spar.neutral_axis')
        # Constraints
        self.driver.add_constraint('Spar.VAL < 1')
        self.driver.add_constraint('Spar.VAG < 1')
        self.driver.add_constraint('Spar.VEL < 1')
        self.driver.add_constraint('Spar.VEG < 1')

if __name__ == "__main__":
    opt_problem = optimizationSpar()
    import time
    tt = time.time()
    opt_problem.run()
    print "\n"
    print opt_problem.neutral_axis
    print "Elapsed time: ", time.time()-tt, "seconds"
