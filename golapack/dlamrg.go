package golapack

import "github.com/whipstein/golinalg/mat"

// Dlamrg will create a permutation list which will merge the elements
// of A (which is composed of two independently sorted sets) into a
// single set which is sorted in ascending order.
func Dlamrg(n1, n2 *int, a *mat.Vector, dtrd1, dtrd2 *int, index *[]int) {
	var i, ind1, ind2, n1sv, n2sv int

	n1sv = (*n1)
	n2sv = (*n2)
	if (*dtrd1) > 0 {
		ind1 = 1
	} else {
		ind1 = (*n1)
	}
	if (*dtrd2) > 0 {
		ind2 = 1 + (*n1)
	} else {
		ind2 = (*n1) + (*n2)
	}
	i = 1
	//     while ( (N1SV > 0) & (N2SV > 0) )
label10:
	;
	if n1sv > 0 && n2sv > 0 {
		if a.Get(ind1-1) <= a.Get(ind2-1) {
			(*index)[i-1] = ind1
			i = i + 1
			ind1 = ind1 + (*dtrd1)
			n1sv = n1sv - 1
		} else {
			(*index)[i-1] = ind2
			i = i + 1
			ind2 = ind2 + (*dtrd2)
			n2sv = n2sv - 1
		}
		goto label10
	}
	//     end while
	if n1sv == 0 {
		for n1sv = 1; n1sv <= n2sv; n1sv++ {
			(*index)[i-1] = ind2
			i = i + 1
			ind2 = ind2 + (*dtrd2)
		}
	} else {
		//     N2SV .EQ. 0
		for n2sv = 1; n2sv <= n1sv; n2sv++ {
			(*index)[i-1] = ind1
			i = i + 1
			ind1 = ind1 + (*dtrd1)
		}
	}
}
