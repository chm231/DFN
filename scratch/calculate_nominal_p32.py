import numpy as np

def get_laxemar_nominal_p32(r_cutoff=1.0, r_max=250.0):
    # Laxemar DFN version 1.2 parameters from Table 2-1
    sets = [
        {'name': 'Set_1', 'P32_total': 1.310, 'type': 'powerlaw', 'kr': 2.85, 'r0': 0.328},
        {'name': 'Set_2', 'P32_total': 1.026, 'type': 'powerlaw', 'kr': 3.04, 'r0': 0.977},
        {'name': 'Set_3', 'P32_total': 0.975, 'type': 'powerlaw', 'kr': 3.01, 'r0': 0.858},
        {'name': 'Set_4', 'P32_total': 2.320, 'type': 'exponential', 'r0': 4.0, 'rmin_true': 0.0}, # exponential from r=0
        {'name': 'Set_5', 'P32_total': 1.400, 'type': 'powerlaw', 'kr': 3.60, 'r0': 0.400}
    ]
    
    results = []
    for s in sets:
        name = s['name']
        P32_t = s['P32_total']
        
        if s['type'] == 'powerlaw':
            kr = s['kr']
            r0 = s['r0']
            rmin = max(r_cutoff, r0)
            pow_val = 2.0 - kr
            
            # Integral of r^2 * PDF on [r0, rmax] is proportional to (rmax^pow - r0^pow) / pow
            if np.abs(pow_val) < 1e-12:
                int_r0 = np.log(r_max) - np.log(r0)
                int_rmin = np.log(r_max) - np.log(rmin)
            else:
                int_r0 = (r_max**pow_val - r0**pow_val) / pow_val
                int_rmin = (r_max**pow_val - rmin**pow_val) / pow_val
                
            ratio = int_rmin / int_r0
            P32_nominal = P32_t * ratio
            
        elif s['type'] == 'exponential':
            lbl = 1.0 / s['r0']
            rmin = r_cutoff
            # Integral of r^2 * exp(-lbl * r)
            int_func = lambda r: -np.exp(-lbl * r) * (r**2 + 2 * r / lbl + 2 / (lbl**2))
            int_r0 = int_func(r_max) - int_func(0.0)
            int_rmin = int_func(r_max) - int_func(rmin)
            
            ratio = int_rmin / int_r0
            P32_nominal = P32_t * ratio
            
        results.append({
            'name': name,
            'P32_total': P32_t,
            'P32_nominal': P32_nominal,
            'ratio': ratio,
            'rmin_used': rmin
        })
        
    return results

if __name__ == '__main__':
    res = get_laxemar_nominal_p32(r_cutoff=1.0)
    print("Set Name | True P32 (Total) | Nominal P32 (r >= 1.0 m) | Reduction Ratio")
    print("-" * 75)
    for r in res:
        print(f"{r['name']:8} | {r['P32_total']:16.3f} | {r['P32_nominal']:24.4f} | {r['ratio']:15.2%}")
