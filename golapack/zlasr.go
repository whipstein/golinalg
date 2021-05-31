package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlasr applies a sequence of real plane rotations to a complex matrix
// A, from either the left or the right.
//
// When SIDE = 'L', the transformation takes the form
//
//    A := P*A
//
// and when SIDE = 'R', the transformation takes the form
//
//    A := A*P**T
//
// where P is an orthogonal matrix consisting of a sequence of z plane
// rotations, with z = M when SIDE = 'L' and z = N when SIDE = 'R',
// and P**T is the transpose of P.
//
// When DIRECT = 'F' (Forward sequence), then
//
//    P = P(z-1) * ... * P(2) * P(1)
//
// and when DIRECT = 'B' (Backward sequence), then
//
//    P = P(1) * P(2) * ... * P(z-1)
//
// where P(k) is a plane rotation matrix defined by the 2-by-2 rotation
//
//    R(k) = (  c(k)  s(k) )
//         = ( -s(k)  c(k) ).
//
// When PIVOT = 'V' (Variable pivot), the rotation is performed
// for the plane (k,k+1), i.e., P(k) has the form
//
//    P(k) = (  1                                            )
//           (       ...                                     )
//           (              1                                )
//           (                   c(k)  s(k)                  )
//           (                  -s(k)  c(k)                  )
//           (                                1              )
//           (                                     ...       )
//           (                                            1  )
//
// where R(k) appears as a rank-2 modification to the identity matrix in
// rows and columns k and k+1.
//
// When PIVOT = 'T' (Top pivot), the rotation is performed for the
// plane (1,k+1), so P(k) has the form
//
//    P(k) = (  c(k)                    s(k)                 )
//           (         1                                     )
//           (              ...                              )
//           (                     1                         )
//           ( -s(k)                    c(k)                 )
//           (                                 1             )
//           (                                      ...      )
//           (                                             1 )
//
// where R(k) appears in rows and columns 1 and k+1.
//
// Similarly, when PIVOT = 'B' (Bottom pivot), the rotation is
// performed for the plane (k,z), giving P(k) the form
//
//    P(k) = ( 1                                             )
//           (      ...                                      )
//           (             1                                 )
//           (                  c(k)                    s(k) )
//           (                         1                     )
//           (                              ...              )
//           (                                     1         )
//           (                 -s(k)                    c(k) )
//
// where R(k) appears in rows and columns k and z.  The rotations are
// performed without ever forming P(k) explicitly.
func Zlasr(side, pivot, direct byte, m, n *int, c, s *mat.Vector, a *mat.CMatrix, lda *int) {
	var temp complex128
	var ctemp, one, stemp, zero float64
	var i, info, j int

	one = 1.0
	zero = 0.0

	//     Test the input parameters
	info = 0
	if !(side == 'L' || side == 'R') {
		info = 1
	} else if !(pivot == 'V' || pivot == 'T' || pivot == 'B') {
		info = 2
	} else if !(direct == 'F' || direct == 'B') {
		info = 3
	} else if (*m) < 0 {
		info = 4
	} else if (*n) < 0 {
		info = 5
	} else if (*lda) < maxint(1, *m) {
		info = 9
	}
	if info != 0 {
		gltest.Xerbla([]byte("ZLASR "), info)
		return
	}

	//     Quick return if possible
	if ((*m) == 0) || ((*n) == 0) {
		return
	}
	if side == 'L' {
		//        Form  P * A
		if pivot == 'V' {
			if direct == 'F' {
				for j = 1; j <= (*m)-1; j++ {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*n); i++ {
							temp = a.Get(j+1-1, i-1)
							a.Set(j+1-1, i-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(j-1, i-1))
							a.Set(j-1, i-1, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(j-1, i-1))
						}
					}
				}
			} else if direct == 'B' {
				for j = (*m) - 1; j >= 1; j-- {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*n); i++ {
							temp = a.Get(j+1-1, i-1)
							a.Set(j+1-1, i-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(j-1, i-1))
							a.Set(j-1, i-1, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(j-1, i-1))
						}
					}
				}
			}
		} else if pivot == 'T' {
			if direct == 'F' {
				for j = 2; j <= (*m); j++ {
					ctemp = c.Get(j - 1 - 1)
					stemp = s.Get(j - 1 - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*n); i++ {
							temp = a.Get(j-1, i-1)
							a.Set(j-1, i-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(0, i-1))
							a.Set(0, i-1, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(0, i-1))
						}
					}
				}
			} else if direct == 'B' {
				for j = (*m); j >= 2; j-- {
					ctemp = c.Get(j - 1 - 1)
					stemp = s.Get(j - 1 - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*n); i++ {
							temp = a.Get(j-1, i-1)
							a.Set(j-1, i-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(0, i-1))
							a.Set(0, i-1, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(0, i-1))
						}
					}
				}
			}
		} else if pivot == 'B' {
			if direct == 'F' {
				for j = 1; j <= (*m)-1; j++ {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*n); i++ {
							temp = a.Get(j-1, i-1)
							a.Set(j-1, i-1, complex(stemp, 0)*a.Get((*m)-1, i-1)+complex(ctemp, 0)*temp)
							a.Set((*m)-1, i-1, complex(ctemp, 0)*a.Get((*m)-1, i-1)-complex(stemp, 0)*temp)
						}
					}
				}
			} else if direct == 'B' {
				for j = (*m) - 1; j >= 1; j-- {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*n); i++ {
							temp = a.Get(j-1, i-1)
							a.Set(j-1, i-1, complex(stemp, 0)*a.Get((*m)-1, i-1)+complex(ctemp, 0)*temp)
							a.Set((*m)-1, i-1, complex(ctemp, 0)*a.Get((*m)-1, i-1)-complex(stemp, 0)*temp)
						}
					}
				}
			}
		}
	} else if side == 'R' {
		//        Form A * P**T
		if pivot == 'V' {
			if direct == 'F' {
				for j = 1; j <= (*n)-1; j++ {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*m); i++ {
							temp = a.Get(i-1, j+1-1)
							a.Set(i-1, j+1-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(i-1, j-1))
							a.Set(i-1, j-1, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(i-1, j-1))
						}
					}
				}
			} else if direct == 'B' {
				for j = (*n) - 1; j >= 1; j-- {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*m); i++ {
							temp = a.Get(i-1, j+1-1)
							a.Set(i-1, j+1-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(i-1, j-1))
							a.Set(i-1, j-1, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(i-1, j-1))
						}
					}
				}
			}
		} else if pivot == 'T' {
			if direct == 'F' {
				for j = 2; j <= (*n); j++ {
					ctemp = c.Get(j - 1 - 1)
					stemp = s.Get(j - 1 - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*m); i++ {
							temp = a.Get(i-1, j-1)
							a.Set(i-1, j-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(i-1, 0))
							a.Set(i-1, 0, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(i-1, 0))
						}
					}
				}
			} else if direct == 'B' {
				for j = (*n); j >= 2; j-- {
					ctemp = c.Get(j - 1 - 1)
					stemp = s.Get(j - 1 - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*m); i++ {
							temp = a.Get(i-1, j-1)
							a.Set(i-1, j-1, complex(ctemp, 0)*temp-complex(stemp, 0)*a.Get(i-1, 0))
							a.Set(i-1, 0, complex(stemp, 0)*temp+complex(ctemp, 0)*a.Get(i-1, 0))
						}
					}
				}
			}
		} else if pivot == 'B' {
			if direct == 'F' {
				for j = 1; j <= (*n)-1; j++ {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*m); i++ {
							temp = a.Get(i-1, j-1)
							a.Set(i-1, j-1, complex(stemp, 0)*a.Get(i-1, (*n)-1)+complex(ctemp, 0)*temp)
							a.Set(i-1, (*n)-1, complex(ctemp, 0)*a.Get(i-1, (*n)-1)-complex(stemp, 0)*temp)
						}
					}
				}
			} else if direct == 'B' {
				for j = (*n) - 1; j >= 1; j-- {
					ctemp = c.Get(j - 1)
					stemp = s.Get(j - 1)
					if (ctemp != one) || (stemp != zero) {
						for i = 1; i <= (*m); i++ {
							temp = a.Get(i-1, j-1)
							a.Set(i-1, j-1, complex(stemp, 0)*a.Get(i-1, (*n)-1)+complex(ctemp, 0)*temp)
							a.Set(i-1, (*n)-1, complex(ctemp, 0)*a.Get(i-1, (*n)-1)-complex(stemp, 0)*temp)
						}
					}
				}
			}
		}
	}
}
