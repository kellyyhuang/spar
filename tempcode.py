self.spar.water_depth=130.
        self.water_depth = 130.
        self.load_condition = 'N'
        self.significant_wave_height = 10.660
        self.significant_wave_period = 13.210
        self.keel_cg_mooring = 51.
        self.keel_cg_operating_system = 20.312
        self.reference_wind_speed = 11.
        self.reference_height = 75.
        self.alpha = 0.110
        self.material_density = 7850.
        self.E = 200.e9
        self.nu = 0.3
        self.yield_stress = 345000000.
        self.rotor_mass = 125000.
        self.tower_mass = 83705.
        self.free_board = 13.
        self.draft = 64.
        self.fixed_ballast_mass = 1244227.77
        self.hull_mass = 890985.086
        self.permanent_ballast_mass = 838450.256
        self.variable_ballast_mass = 418535.462
        self.number_of_sections = 4
        self.outer_diameter = [5., 6., 6., 9.]
        self.length = [6., 12., 15., 44.]
        self.end_elevation = [7., -5., -20., -64.]
        self.start_elevation = [13., 7., -5., -20.]
        self.bulk_head = ['N', 'T', 'N', 'B']


if 
    opt_problem = optimizationSpar()
    import time
    tt = time.time()
    opt_problem.run()
    print "\n"
    print 'neutral axis:'
    print opt_problem.spar.neutral_axis
    print 'wall thickness:'
    print opt_problem.spar.wall_thickness
    print 'mass:'
    print opt_problem.spar.shell_and_ring_mass
    print 'unity checks:'
    print opt_problem.spar.VAL
    print opt_problem.spar.VAG
    print opt_problem.spar.VEL
    print opt_problem.spar.VEG
    print 'number of rings:'
    print opt_problem.spar.number_of_rings
    print 'compactness checks:'
    print opt_problem.spar.flange_compactness
    print opt_problem.spar.web_compactness
    print "Elapsed time: ", time.time()-tt, "seconds"
    yna = convert_units(opt_problem.spar.neutral_axis ,'m','inch')
    filteredStiffeners = filtered_stiffeners_table()
    for i in range (0,len(filteredStiffeners)-1):
        stiffener_bef = filteredStiffeners[i]
        stiffener_aft = filteredStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            index = i+1
    print index
    example = Spar()
    example.wall_thickness = opt_problem.spar.wall_thickness
    example.number_of_rings = opt_problem.spar.number_of_rings
    example.stiffener_index = index
    example.initial_pass = False
    example.run()
    print example.VAL
    print example.VAG
    print example.VEL
    print example.VEG
    print example.shell_and_ring_mass

    unity = max(max(example.VAL),max(example.VAG),max(example.VEL),max(example.VEG))   
    while ((unity-1.0) > 1e-6):
        index += 1
        example.stiffener_index = index
        example.run()
        unity = max(max(example.VAL),max(example.VAG),max(example.VEL),max(example.VEG)) 
        print unity-1.0
    print example.stiffener_index
    print example.VAL
    print example.VAG
    print example.VEL
    print example.VEG
    print example.shell_and_ring_mass


 second_fit = Spar()
    example.wall_thickness = opt_problem.spar.wall_thickness
    example.number_of_rings = opt_problem.spar.number_of_rings
    example.stiffener_index = index
    example.initial_pass = False
    example.run()
    unity = max(max(example.VAL),max(example.VAG),max(example.VEL),max(example.VEG))   
    print unity







    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-6):
        if index <125:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
            print unity-1.0
        else:
            second_fit.stiffener_index = opt_index
            section_to_increase = second_fit.number_of_rings.index(max(VAL))  
            second_fit.number_of_rings[section_to_increase] += 1 
       

    print second_fit.stiffener_index
    print second_fit.VAL
    print second_fit.VAG
    print second_fit.VEL
    print second_fit.VEG
    print second_fit.shell_and_ring_mass



















    yna = convert_units(example.spar.neutral_axis ,'m','inch')
    filteredStiffeners = filtered_stiffeners_table()
    for i in range (0,len(filteredStiffeners)-1):
        stiffener_bef = filteredStiffeners[i]
        stiffener_aft = filteredStiffeners[i+1]
        if yna > stiffener_bef[6] and yna<stiffener_aft[6]:
            opt_index = i+1
    #print opt_index
    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False

    second_fit.water_depth =  example.water_depth
    second_fit.load_condition = example.load_condition
    second_fit.significant_wave_height =  example.significant_wave_height
    second_fit.significant_wave_period = example.significant_wave_period
    second_fit.keel_cg_mooring = example.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.keel_cg_operating_system
    second_fit.reference_wind_speed = example.reference_wind_speed
    second_fit.reference_height = example.reference_height
    second_fit.alpha = example.alpha
    second_fit.material_density = example.material_density
    second_fit.E = example.E
    second_fit.nu =example.nu
    second_fit.yield_stress = example.yield_stress
    second_fit.rotor_mass = example.rotor_mass
    second_fit.tower_mass = example.tower_mass
    second_fit.free_board = example.free_board
    second_fit.draft = example.draft
    second_fit.fixed_ballast_mass = example.fixed_ballast_mass
    second_fit.hull_mass = example.hull_mass
    second_fit.permanent_ballast_mass = example.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.variable_ballast_mass
    second_fit.number_of_sections = example.number_of_sections
    second_fit.outer_diameter = example.outer_diameter
    second_fit.length = example.length
    second_fit.end_elevation = example.end_elevation
    second_fit.start_elevation = example.start_elevation
    second_fit.bulk_head = example.bulk_head
    second_fit.system_acceleration=example.system_acceleration
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
            #print unity-1.0
        else:
            second_fit.stiffener_index = opt_index
            for i in range(0,second_fit.number_of_sections):
                if second_fit.VAL[i] >1. or second_fit.VAG[i]>1. or second_fit.VEL[i]>1. or second_fit.VEG[i]>1.:    
                    second_fit.number_of_rings[i] += 1 
                    second_fit.run()
                    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
                    #print second_fit.number_of_rings
    print 'water depth: ', second_fit.water_depth 
    print 'number of stiffeners: ',second_fit.number_of_rings
    print 'stiffener: ', filteredStiffeners[second_fit.stiffener_index]
    print 'VAL: ',second_fit.VAL
    print 'VAG: ',second_fit.VAG
    print 'VEL: ',second_fit.VEL
    print 'VEG: ',second_fit.VEG
    print 'wall thickness: ',second_fit.wall_thickness
    print 'shell+ring+bulkhead mass: ',second_fit.shell_ring_bulkhead_mass
    print "Elapsed time: ", time.time()-tt, " seconds"












    #print opt_index
    example.initial_pass = False
    example.stiffener_index = opt_index
    example.run()

    index = opt_index
    unity = max(max(example.VAL),max(example.VAG),max(example.VEL),max(example.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            example.stiffener_index = index
            example.run()
            unity = max(max(example.VAL),max(example.VAG),max(example.VEL),max(example.VEG)) 
            #print unity-1.0
        else:
            example.stiffener_index = opt_index
            for i in range(0,example.number_of_sections):
                if example.VAL[i] >1. or example.VAG[i]>1. or example.VEL[i]>1. or example.VEG[i]>1.:    
                    example.number_of_rings[i] += 1 
                    example.run()
                    unity = max(max(example.VAL),max(example.VAG),max(example.VEL),max(example.VEG)) 
                    #print second_fit.number_of_rings
    print 'water depth: ', example.water_depth 
    print 'number of stiffeners: ', example.number_of_rings
    print 'stiffener: ', filteredStiffeners[example.stiffener_index]
    print 'VAL: ',example.VAL
    print 'VAG: ',example.VAG
    print 'VEL: ',example.VEL
    print 'VEG: ',example.VEG
    print 'wall thickness: ',example.wall_thickness
    print 'shell+ring+bulkhead mass: ',example.shell_ring_bulkhead_mass
    print "Elapsed time: ", time.time()-tt, " seconds"



    second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False

    second_fit.water_depth =  example.water_depth
    second_fit.load_condition = example.load_condition
    second_fit.significant_wave_height =  example.significant_wave_height
    second_fit.significant_wave_period = example.significant_wave_period
    second_fit.keel_cg_mooring = example.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.keel_cg_operating_system
    second_fit.reference_wind_speed = example.reference_wind_speed
    second_fit.reference_height = example.reference_height
    second_fit.alpha = example.alpha
    second_fit.material_density = example.material_density
    second_fit.E = example.E
    second_fit.nu =example.nu
    second_fit.yield_stress = example.yield_stress
    second_fit.rotor_mass = example.rotor_mass
    second_fit.tower_mass = example.tower_mass
    second_fit.free_board = example.free_board
    second_fit.draft = example.draft
    second_fit.fixed_ballast_mass = example.fixed_ballast_mass
    second_fit.hull_mass = example.hull_mass
    second_fit.permanent_ballast_mass = example.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.variable_ballast_mass
    second_fit.number_of_sections = example.number_of_sections
    second_fit.outer_diameter = example.outer_diameter
    second_fit.length = example.length
    second_fit.end_elevation = example.end_elevation
    second_fit.start_elevation = example.start_elevation
    second_fit.bulk_head = example.bulk_head
    second_fit.system_acceleration=example.system_acceleration
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        else:
            second_fit.stiffener_index = opt_index
            for i in range(0,second_fit.number_of_sections):
                if second_fit.VAL[i] >1. or second_fit.VAG[i]>1. or second_fit.VEL[i]>1. or second_fit.VEG[i]>1.:    
                    second_fit.number_of_rings[i] += 1 
                    second_fit.run()
                    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 

















second_fit = Spar()
    second_fit.wall_thickness = example.spar.wall_thickness
    second_fit.number_of_rings = example.spar.number_of_rings
    second_fit.stiffener_index = opt_index
    second_fit.initial_pass = False

    second_fit.water_depth =  example.water_depth
    second_fit.load_condition = example.load_condition
    second_fit.significant_wave_height =  example.significant_wave_height
    second_fit.significant_wave_period = example.significant_wave_period
    second_fit.keel_cg_mooring = example.keel_cg_mooring
    second_fit.keel_cg_operating_system = example.keel_cg_operating_system
    second_fit.reference_wind_speed = example.reference_wind_speed
    second_fit.reference_height = example.reference_height
    second_fit.alpha = example.alpha
    second_fit.material_density = example.material_density
    second_fit.E = example.E
    second_fit.nu =example.nu
    second_fit.yield_stress = example.yield_stress
    second_fit.rotor_mass = example.rotor_mass
    second_fit.tower_mass = example.tower_mass
    second_fit.free_board = example.free_board
    second_fit.draft = example.draft
    second_fit.fixed_ballast_mass = example.fixed_ballast_mass
    second_fit.hull_mass = example.hull_mass
    second_fit.permanent_ballast_mass = example.permanent_ballast_mass
    second_fit.variable_ballast_mass = example.variable_ballast_mass
    second_fit.number_of_sections = example.number_of_sections
    second_fit.outer_diameter = example.outer_diameter
    second_fit.length = example.length
    second_fit.end_elevation = example.end_elevation
    second_fit.start_elevation = example.start_elevation
    second_fit.bulk_head = example.bulk_head
    second_fit.system_acceleration=example.system_acceleration
    second_fit.run()
    index = opt_index
    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG))   
    while ((unity-1.0) > 1e-7):
        if index <124:
            index += 1
            second_fit.stiffener_index = index
            second_fit.run()
            unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 
        else:
            second_fit.stiffener_index = opt_index
            for i in range(0,second_fit.number_of_sections):
                if second_fit.VAL[i] >1. or second_fit.VAG[i]>1. or second_fit.VEL[i]>1. or second_fit.VEG[i]>1.:    
                    second_fit.number_of_rings[i] += 1 
                    second_fit.run()
                    unity = max(max(second_fit.VAL),max(second_fit.VAG),max(second_fit.VEL),max(second_fit.VEG)) 