def kg_to_lb(wt):
    return  wt / 0.45

def lb_to_kg(wt):
    return wt * 0.45

def find_max(lst):
   max = lst[0]
   for n in lst:
        if n > max:
            max = n
   return max
