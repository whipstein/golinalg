package golapack

import (
	"github.com/whipstein/golinalg/mat"
)

// Dlarft forms the triangular factor T of a real block reflector H
// of order n, which is defined as a product of k elementary reflectors.
//
// If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
//
// If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
//
// If STOREV = 'C', the vector which defines the elementary reflector
// H(i) is stored in the i-th column of the array V, and
//
//    H  =  I - V * T * V**T
//
// If STOREV = 'R', the vector which defines the elementary reflector
// H(i) is stored in the i-th row of the array V, and
//
//    H  =  I - V**T * T * V
func Dlarft(direct, storev byte, n, k int, v *mat.Matrix, tau *mat.Vector, t *mat.Matrix) {
	var one, zero float64
	var i, j, lastv, prevlastv int
	var err error

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if n == 0 {
		return
	}

	if direct == 'F' {
		prevlastv = n
		for i = 1; i <= k; i++ {
			prevlastv = max(i, prevlastv)
			if tau.Get(i-1) == zero {
				//              H(i)  =  I
				for j = 1; j <= i; j++ {
					t.Set(j-1, i-1, zero)
				}
			} else {
				//              general case
				if storev == 'C' {
					//                 Skip any trailing zeros.
					for lastv = n; lastv >= i+1; lastv-- {
						if v.Get(lastv-1, i-1) != zero {
							break
						}
					}
					for j = 1; j <= i-1; j++ {
						t.Set(j-1, i-1, -tau.Get(i-1)*v.Get(i-1, j-1))
					}
					j = min(lastv, prevlastv)

					//                 T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)**T * V(i:j,i)
					if err = t.Off(0, i-1).Vector().Gemv(Trans, j-i, i-1, -tau.Get(i-1), v.Off(i, 0), v.Off(i, i-1).Vector(), 1, one, 1); err != nil {
						panic(err)
					}
				} else {
					//                 Skip any trailing zeros.
					for lastv = n; lastv >= i+1; lastv-- {
						if v.Get(i-1, lastv-1) != zero {
							break
						}
					}
					for j = 1; j <= i-1; j++ {
						t.Set(j-1, i-1, -tau.Get(i-1)*v.Get(j-1, i-1))
					}
					j = min(lastv, prevlastv)

					//                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)**T
					if err = t.Off(0, i-1).Vector().Gemv(NoTrans, i-1, j-i, -tau.Get(i-1), v.Off(0, i), v.Off(i-1, i).Vector(), v.Rows, one, 1); err != nil {
						panic(err)
					}
				}

				//              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
				if err = t.Off(0, i-1).Vector().Trmv(Upper, NoTrans, NonUnit, i-1, t, 1); err != nil {
					panic(err)
				}
				t.Set(i-1, i-1, tau.Get(i-1))
				if i > 1 {
					prevlastv = max(prevlastv, lastv)
				} else {
					prevlastv = lastv
				}
			}
		}
	} else {
		prevlastv = 1
		for i = k; i >= 1; i-- {
			if tau.Get(i-1) == zero {
				//              H(i)  =  I
				for j = i; j <= k; j++ {
					t.Set(j-1, i-1, zero)
				}
			} else {
				//              general case
				if i < k {
					if storev == 'C' {
						//                    Skip any leading zeros.
						for lastv = 1; lastv <= i-1; lastv++ {
							if v.Get(lastv-1, i-1) != zero {
								break
							}
						}
						for j = i + 1; j <= k; j++ {
							t.Set(j-1, i-1, -tau.Get(i-1)*v.Get(n-k+i-1, j-1))
						}
						j = max(lastv, prevlastv)

						//                    T(i+1:k,i) = -tau(i) * V(j:n-k+i,i+1:k)**T * V(j:n-k+i,i)
						if err = t.Off(i, i-1).Vector().Gemv(Trans, n-k+i-j, k-i, -tau.Get(i-1), v.Off(j-1, i), v.Off(j-1, i-1).Vector(), 1, one, 1); err != nil {
							panic(err)
						}
					} else {
						//                    Skip any leading zeros.
						for lastv = 1; lastv <= i-1; lastv++ {
							if v.Get(i-1, lastv-1) != zero {
								break
							}
						}
						for j = i + 1; j <= k; j++ {
							t.Set(j-1, i-1, -tau.Get(i-1)*v.Get(j-1, n-k+i-1))
						}
						j = max(lastv, prevlastv)

						//                    T(i+1:k,i) = -tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)**T
						if err = t.Off(i, i-1).Vector().Gemv(NoTrans, k-i, n-k+i-j, -tau.Get(i-1), v.Off(i, j-1), v.Off(i-1, j-1).Vector(), v.Rows, one, 1); err != nil {
							panic(err)
						}
					}

					//                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i)
					if err = t.Off(i, i-1).Vector().Trmv(Lower, NoTrans, NonUnit, k-i, t.Off(i, i), 1); err != nil {
						panic(err)
					}
					if i > 1 {
						prevlastv = min(prevlastv, lastv)
					} else {
						prevlastv = lastv
					}
				}
				t.Set(i-1, i-1, tau.Get(i-1))
			}
		}
	}
}
