package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Zlarft forms the triangular factor T of a complex block reflector H
// of order n, which is defined as a product of k elementary reflectors.
//
// If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
//
// If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
//
// If STOREV = 'C', the vector which defines the elementary reflector
// H(i) is stored in the i-th column of the array V, and
//
//    H  =  I - V * T * V**H
//
// If STOREV = 'R', the vector which defines the elementary reflector
// H(i) is stored in the i-th row of the array V, and
//
//    H  =  I - V**H * T * V
func Zlarft(direct, storev byte, n, k *int, v *mat.CMatrix, ldv *int, tau *mat.CVector, t *mat.CMatrix, ldt *int) {
	var one, zero complex128
	var i, j, lastv, prevlastv int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if direct == 'F' {
		prevlastv = (*n)
		for i = 1; i <= (*k); i++ {
			prevlastv = maxint(prevlastv, i)
			if tau.Get(i-1) == zero {
				//              H(i)  =  I
				for j = 1; j <= i; j++ {
					t.Set(j-1, i-1, zero)
				}
			} else {
				//              general case
				if storev == 'C' {
					//                 Skip any trailing zeros.
					for lastv = (*n); lastv >= i+1; lastv-- {
						if v.Get(lastv-1, i-1) != zero {
							break
						}
					}
					for j = 1; j <= i-1; j++ {
						t.Set(j-1, i-1, -tau.Get(i-1)*v.GetConj(i-1, j-1))
					}
					j = minint(lastv, prevlastv)

					//                 T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)**H * V(i:j,i)
					goblas.Zgemv(ConjTrans, toPtr(j-i), toPtr(i-1), toPtrc128(-tau.Get(i-1)), v.Off(i+1-1, 0), ldv, v.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), &one, t.CVector(0, i-1), func() *int { y := 1; return &y }())
				} else {
					//                 Skip any trailing zeros.
					for lastv = (*n); lastv >= i+1; lastv-- {
						if v.Get(i-1, lastv-1) != zero {
							break
						}
					}
					for j = 1; j <= i-1; j++ {
						t.Set(j-1, i-1, -tau.Get(i-1)*v.Get(j-1, i-1))
					}
					j = minint(lastv, prevlastv)

					//                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)**H
					goblas.Zgemm(NoTrans, ConjTrans, toPtr(i-1), func() *int { y := 1; return &y }(), toPtr(j-i), toPtrc128(-tau.Get(i-1)), v.Off(0, i+1-1), ldv, v.Off(i-1, i+1-1), ldv, &one, t.Off(0, i-1), ldt)
				}

				//              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
				goblas.Ztrmv(Upper, NoTrans, NonUnit, toPtr(i-1), t, ldt, t.CVector(0, i-1), func() *int { y := 1; return &y }())
				t.Set(i-1, i-1, tau.Get(i-1))
				if i > 1 {
					prevlastv = maxint(prevlastv, lastv)
				} else {
					prevlastv = lastv
				}
			}
		}
	} else {
		prevlastv = 1
		for i = (*k); i >= 1; i-- {
			if tau.Get(i-1) == zero {
				//              H(i)  =  I
				for j = i; j <= (*k); j++ {
					t.Set(j-1, i-1, zero)
				}
			} else {
				//              general case
				if i < (*k) {
					if storev == 'C' {
						//                    Skip any leading zeros.
						for lastv = 1; lastv <= i-1; lastv++ {
							if v.Get(lastv-1, i-1) != zero {
								break
							}
						}
						for j = i + 1; j <= (*k); j++ {
							t.Set(j-1, i-1, -tau.Get(i-1)*v.GetConj((*n)-(*k)+i-1, j-1))
						}
						j = maxint(lastv, prevlastv)

						//                    T(i+1:k,i) = -tau(i) * V(j:n-k+i,i+1:k)**H * V(j:n-k+i,i)
						goblas.Zgemv(ConjTrans, toPtr((*n)-(*k)+i-j), toPtr((*k)-i), toPtrc128(-tau.Get(i-1)), v.Off(j-1, i+1-1), ldv, v.CVector(j-1, i-1), func() *int { y := 1; return &y }(), &one, t.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
					} else {
						//                    Skip any leading zeros.
						for lastv = 1; lastv <= i-1; lastv++ {
							if v.Get(i-1, lastv-1) != zero {
								break
							}
						}
						for j = i + 1; j <= (*k); j++ {
							t.Set(j-1, i-1, -tau.Get(i-1)*v.Get(j-1, (*n)-(*k)+i-1))
						}
						j = maxint(lastv, prevlastv)

						//                    T(i+1:k,i) = -tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)**H
						goblas.Zgemm(NoTrans, ConjTrans, toPtr((*k)-i), func() *int { y := 1; return &y }(), toPtr((*n)-(*k)+i-j), toPtrc128(-tau.Get(i-1)), v.Off(i+1-1, j-1), ldv, v.Off(i-1, j-1), ldv, &one, t.Off(i+1-1, i-1), ldt)
					}

					//                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i)
					goblas.Ztrmv(Lower, NoTrans, NonUnit, toPtr((*k)-i), t.Off(i+1-1, i+1-1), ldt, t.CVector(i+1-1, i-1), func() *int { y := 1; return &y }())
					if i > 1 {
						prevlastv = minint(prevlastv, lastv)
					} else {
						prevlastv = lastv
					}
				}
				t.Set(i-1, i-1, tau.Get(i-1))
			}
		}
	}
}
