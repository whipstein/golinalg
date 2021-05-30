package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dlasrt Sort the numbers in D in increasing order (if ID = 'I') or
// in decreasing order (if ID = 'D' ).
//
// Use Quick Sort, reverting to Insertion sort on arrays of
// size <= 20. Dimension of STACK limits N to about 2**32.
func Dlasrt(id byte, n *int, d *mat.Vector, info *int) {
	var d1, d2, d3, dmnmx, tmp float64
	var dir, endd, i, j, _select, start, stkpnt int
	stack := make([]int, 2*32)

	_select = 20

	//     Test the input parameters.
	(*info) = 0
	dir = -1
	if id == 'D' {
		dir = 0
	} else if id == 'I' {
		dir = 1
	}
	if dir == -1 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASRT"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}

	stkpnt = 1
	stack[0+(0)*2] = 1
	stack[1+(0)*2] = (*n)
label10:
	;
	start = stack[0+(stkpnt-1)*2]
	endd = stack[1+(stkpnt-1)*2]
	stkpnt = stkpnt - 1
	if endd-start <= _select && endd-start > 0 {
		//        Do Insertion sort on D( START:ENDD )
		if dir == 0 {
			//           Sort into decreasing order
			for i = start + 1; i <= endd; i++ {
				for j = i; j >= start+1; j-- {
					if d.Get(j-1) > d.Get(j-1-1) {
						dmnmx = d.Get(j - 1)
						d.Set(j-1, d.Get(j-1-1))
						d.Set(j-1-1, dmnmx)
					} else {
						goto label30
					}
				}
			label30:
			}

		} else {
			//           Sort into increasing order
			for i = start + 1; i <= endd; i++ {
				for j = i; j >= start+1; j-- {
					if d.Get(j-1) < d.Get(j-1-1) {
						dmnmx = d.Get(j - 1)
						d.Set(j-1, d.Get(j-1-1))
						d.Set(j-1-1, dmnmx)
					} else {
						goto label50
					}
				}
			label50:
			}

		}

	} else if endd-start > _select {
		//        Partition D( START:ENDD ) and stack parts, largest one first
		//
		//        Choose partition entry as median of 3
		d1 = d.Get(start - 1)
		d2 = d.Get(endd - 1)
		i = (start + endd) / 2
		d3 = d.Get(i - 1)
		if d1 < d2 {
			if d3 < d1 {
				dmnmx = d1
			} else if d3 < d2 {
				dmnmx = d3
			} else {
				dmnmx = d2
			}
		} else {
			if d3 < d2 {
				dmnmx = d2
			} else if d3 < d1 {
				dmnmx = d3
			} else {
				dmnmx = d1
			}
		}

		if dir == 0 {
			//           Sort into decreasing order
			i = start - 1
			j = endd + 1
		label60:
			;
		label70:
			;
			j = j - 1
			if d.Get(j-1) < dmnmx {
				goto label70
			}
		label80:
			;
			i = i + 1
			if d.Get(i-1) > dmnmx {
				goto label80
			}
			if i < j {
				tmp = d.Get(i - 1)
				d.Set(i-1, d.Get(j-1))
				d.Set(j-1, tmp)
				goto label60
			}
			if j-start > endd-j-1 {
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = start
				stack[1+(stkpnt-1)*2] = j
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = j + 1
				stack[1+(stkpnt-1)*2] = endd
			} else {
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = j + 1
				stack[1+(stkpnt-1)*2] = endd
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = start
				stack[1+(stkpnt-1)*2] = j
			}
		} else {
			//           Sort into increasing order
			i = start - 1
			j = endd + 1
		label90:
			;
		label100:
			;
			j = j - 1
			if d.Get(j-1) > dmnmx {
				goto label100
			}
		label110:
			;
			i = i + 1
			if d.Get(i-1) < dmnmx {
				goto label110
			}
			if i < j {
				tmp = d.Get(i - 1)
				d.Set(i-1, d.Get(j-1))
				d.Set(j-1, tmp)
				goto label90
			}
			if j-start > endd-j-1 {
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = start
				stack[1+(stkpnt-1)*2] = j
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = j + 1
				stack[1+(stkpnt-1)*2] = endd
			} else {
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = j + 1
				stack[1+(stkpnt-1)*2] = endd
				stkpnt = stkpnt + 1
				stack[0+(stkpnt-1)*2] = start
				stack[1+(stkpnt-1)*2] = j
			}
		}
	}
	if stkpnt > 0 {
		goto label10
	}
}
