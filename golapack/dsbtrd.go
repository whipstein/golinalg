package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbtrd reduces a real symmetric band matrix A to symmetric
// tridiagonal form T by an orthogonal similarity transformation:
// Q**T * A * Q = T.
func Dsbtrd(vect, uplo byte, n, kd *int, ab *mat.Matrix, ldab *int, d, e *mat.Vector, q *mat.Matrix, ldq *int, work *mat.Vector, info *int) {
	var initq, upper, wantq bool
	var one, temp, zero float64
	var i, i2, ibl, inca, incx, iqaend, iqb, iqend, j, j1, j1end, j1inc, j2, jend, jin, jinc, k, kd1, kdm1, kdn, l, last, lend, nq, nr, nrt int

	zero = 0.0
	one = 1.0

	//     Test the input parameters
	initq = vect == 'V'
	wantq = initq || vect == 'U'
	upper = uplo == 'U'
	kd1 = (*kd) + 1
	kdm1 = (*kd) - 1
	incx = (*ldab) - 1
	iqend = 1

	(*info) = 0
	if !wantq && vect != 'N' {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*kd) < 0 {
		(*info) = -4
	} else if (*ldab) < kd1 {
		(*info) = -6
	} else if (*ldq) < maxint(1, *n) && wantq {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSBTRD"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Initialize Q to the unit matrix, if needed
	if initq {
		Dlaset('F', n, n, &zero, &one, q, ldq)
	}

	//     Wherever possible, plane rotations are generated and applied in
	//     vector operations of length NR over the index set J1:J2:KD1.
	//
	//     The cosines and sines of the plane rotations are stored in the
	//     arrays D and WORK.
	inca = kd1 * (*ldab)
	kdn = minint((*n)-1, *kd)
	if upper {
		if (*kd) > 1 {
			//           Reduce to tridiagonal form, working with upper triangle
			nr = 0
			j1 = kdn + 2
			j2 = 1

			for i = 1; i <= (*n)-2; i++ {
				//              Reduce i-th row of matrix to tridiagonal form
				for k = kdn + 1; k >= 2; k-- {
					j1 = j1 + kdn
					j2 = j2 + kdn

					if nr > 0 {
						//                    generate plane rotations to annihilate nonzero
						//                    elements which have been created outside the band
						Dlargv(&nr, ab.Vector(0, j1-1-1), &inca, work.Off(j1-1), &kd1, d.Off(j1-1), &kd1)

						//                    apply rotations from the right
						//
						//
						//                    Dependent on the the number of diagonals either
						//                    DLARTV or DROT is used
						if nr >= 2*(*kd)-1 {
							for l = 1; l <= (*kd)-1; l++ {
								Dlartv(&nr, ab.Vector(l+1-1, j1-1-1), &inca, ab.Vector(l-1, j1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
							}

						} else {
							jend = j1 + (nr-1)*kd1
							for _, jinc = range genIter(j1, jend, kd1) {
								goblas.Drot(&kdm1, ab.Vector(1, jinc-1-1), func() *int { y := 1; return &y }(), ab.Vector(0, jinc-1), func() *int { y := 1; return &y }(), d.GetPtr(jinc-1), work.GetPtr(jinc-1))
							}
						}
					}

					if k > 2 {
						if k <= (*n)-i+1 {
							//                       generate plane rotation to annihilate a(i,i+k-1)
							//                       within the band
							Dlartg(ab.GetPtr((*kd)-k+3-1, i+k-2-1), ab.GetPtr((*kd)-k+2-1, i+k-1-1), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1), &temp)
							ab.Set((*kd)-k+3-1, i+k-2-1, temp)

							//                       apply rotation from the right
							goblas.Drot(toPtr(k-3), ab.Vector((*kd)-k+4-1, i+k-2-1), func() *int { y := 1; return &y }(), ab.Vector((*kd)-k+3-1, i+k-1-1), func() *int { y := 1; return &y }(), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1))
						}
						nr = nr + 1
						j1 = j1 - kdn - 1
					}

					//                 apply plane rotations from both sides to diagonal
					//                 blocks
					if nr > 0 {
						Dlar2v(&nr, ab.Vector(kd1-1, j1-1-1), ab.Vector(kd1-1, j1-1), ab.Vector((*kd)-1, j1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
					}

					//                 apply plane rotations from the left
					if nr > 0 {
						if 2*(*kd)-1 < nr {
							//                    Dependent on the the number of diagonals either
							//                    DLARTV or DROT is used
							for l = 1; l <= (*kd)-1; l++ {
								if j2+l > (*n) {
									nrt = nr - 1
								} else {
									nrt = nr
								}
								if nrt > 0 {
									Dlartv(&nrt, ab.Vector((*kd)-l-1, j1+l-1), &inca, ab.Vector((*kd)-l+1-1, j1+l-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
								}
							}
						} else {
							j1end = j1 + kd1*(nr-2)
							if j1end >= j1 {
								for jin = j1; jin <= j1end; jin += kd1 {
									goblas.Drot(toPtr((*kd)-1), ab.Vector((*kd)-1-1, jin+1-1), &incx, ab.Vector((*kd)-1, jin+1-1), &incx, d.GetPtr(jin-1), work.GetPtr(jin-1))
								}
							}
							lend = minint(kdm1, (*n)-j2)
							last = j1end + kd1
							if lend > 0 {
								goblas.Drot(&lend, ab.Vector((*kd)-1-1, last+1-1), &incx, ab.Vector((*kd)-1, last+1-1), &incx, d.GetPtr(last-1), work.GetPtr(last-1))
							}
						}
					}

					if wantq {
						//                    accumulate product of plane rotations in Q
						if initq {
							//                 take advantage of the fact that Q was
							//                 initially the Identity matrix
							iqend = maxint(iqend, j2)
							i2 = maxint(0, k-3)
							iqaend = 1 + i*(*kd)
							if k == 2 {
								iqaend = iqaend + (*kd)
							}
							iqaend = minint(iqaend, iqend)
							for j = j1; j <= j2; j += kd1 {
								ibl = i - i2/kdm1
								i2 = i2 + 1
								iqb = maxint(1, j-ibl)
								nq = 1 + iqaend - iqb
								iqaend = minint(iqaend+(*kd), iqend)
								goblas.Drot(&nq, q.Vector(iqb-1, j-1-1), func() *int { y := 1; return &y }(), q.Vector(iqb-1, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), work.GetPtr(j-1))
							}
						} else {

							for j = j1; j <= j2; j += kd1 {
								goblas.Drot(n, q.Vector(0, j-1-1), func() *int { y := 1; return &y }(), q.Vector(0, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), work.GetPtr(j-1))
							}
						}

					}

					if j2+kdn > (*n) {
						//                    adjust J2 to keep within the bounds of the matrix
						nr = nr - 1
						j2 = j2 - kdn - 1
					}

					for j = j1; j <= j2; j += kd1 {
						//                    create nonzero element a(j-1,j+kd) outside the band
						//                    and store it in WORK
						work.Set(j+(*kd)-1, work.Get(j-1)*ab.Get(0, j+(*kd)-1))
						ab.Set(0, j+(*kd)-1, d.Get(j-1)*ab.Get(0, j+(*kd)-1))
					}
				}
			}
		}

		if (*kd) > 0 {
			//           copy off-diagonal elements to E
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, ab.Get((*kd)-1, i+1-1))
			}
		} else {
			//           set E to zero if original matrix was diagonal
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, zero)
			}
		}

		//        copy diagonal elements to D
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, ab.Get(kd1-1, i-1))
		}

	} else {

		if (*kd) > 1 {
			//           Reduce to tridiagonal form, working with lower triangle
			nr = 0
			j1 = kdn + 2
			j2 = 1

			for i = 1; i <= (*n)-2; i++ {
				//              Reduce i-th column of matrix to tridiagonal form
				for k = kdn + 1; k >= 2; k-- {
					j1 = j1 + kdn
					j2 = j2 + kdn

					if nr > 0 {
						//                    generate plane rotations to annihilate nonzero
						//                    elements which have been created outside the band
						Dlargv(&nr, ab.Vector(kd1-1, j1-kd1-1), &inca, work.Off(j1-1), &kd1, d.Off(j1-1), &kd1)

						//                    apply plane rotations from one side
						//
						//
						//                    Dependent on the the number of diagonals either
						//                    DLARTV or DROT is used
						if nr > 2*(*kd)-1 {
							for l = 1; l <= (*kd)-1; l++ {
								Dlartv(&nr, ab.Vector(kd1-l-1, j1-kd1+l-1), &inca, ab.Vector(kd1-l+1-1, j1-kd1+l-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
							}
						} else {
							jend = j1 + kd1*(nr-1)
							for jinc = j1; jinc <= jend; jinc += kd1 {
								goblas.Drot(&kdm1, ab.Vector((*kd)-1, jinc-(*kd)-1), &incx, ab.Vector(kd1-1, jinc-(*kd)-1), &incx, d.GetPtr(jinc-1), work.GetPtr(jinc-1))
							}
						}

					}

					if k > 2 {
						if k <= (*n)-i+1 {
							//                       generate plane rotation to annihilate a(i+k-1,i)
							//                       within the band
							Dlartg(ab.GetPtr(k-1-1, i-1), ab.GetPtr(k-1, i-1), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1), &temp)
							ab.Set(k-1-1, i-1, temp)

							//                       apply rotation from the left
							goblas.Drot(toPtr(k-3), ab.Vector(k-2-1, i+1-1), toPtr((*ldab)-1), ab.Vector(k-1-1, i+1-1), toPtr((*ldab)-1), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1))
						}
						nr = nr + 1
						j1 = j1 - kdn - 1
					}

					//                 apply plane rotations from both sides to diagonal
					//                 blocks
					if nr > 0 {
						Dlar2v(&nr, ab.Vector(0, j1-1-1), ab.Vector(0, j1-1), ab.Vector(1, j1-1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
					}

					//                 apply plane rotations from the right
					//
					//
					//                    Dependent on the the number of diagonals either
					//                    DLARTV or DROT is used
					if nr > 0 {
						if nr > 2*(*kd)-1 {
							for l = 1; l <= (*kd)-1; l++ {
								if j2+l > (*n) {
									nrt = nr - 1
								} else {
									nrt = nr
								}
								if nrt > 0 {
									Dlartv(&nrt, ab.Vector(l+2-1, j1-1-1), &inca, ab.Vector(l+1-1, j1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
								}
							}
						} else {
							j1end = j1 + kd1*(nr-2)
							if j1end >= j1 {
								for j1inc = j1; j1inc <= j1end; j1inc += kd1 {
									goblas.Drot(&kdm1, ab.Vector(2, j1inc-1-1), func() *int { y := 1; return &y }(), ab.Vector(1, j1inc-1), func() *int { y := 1; return &y }(), d.GetPtr(j1inc-1), work.GetPtr(j1inc-1))
								}
							}
							lend = minint(kdm1, (*n)-j2)
							last = j1end + kd1
							if lend > 0 {
								goblas.Drot(&lend, ab.Vector(2, last-1-1), func() *int { y := 1; return &y }(), ab.Vector(1, last-1), func() *int { y := 1; return &y }(), d.GetPtr(last-1), work.GetPtr(last-1))
							}
						}
					}

					if wantq {
						//                    accumulate product of plane rotations in Q
						if initq {
							//                 take advantage of the fact that Q was
							//                 initially the Identity matrix
							iqend = maxint(iqend, j2)
							i2 = maxint(0, k-3)
							iqaend = 1 + i*(*kd)
							if k == 2 {
								iqaend = iqaend + (*kd)
							}
							iqaend = minint(iqaend, iqend)
							for j = j1; j <= j2; j += kd1 {
								ibl = i - i2/kdm1
								i2 = i2 + 1
								iqb = maxint(1, j-ibl)
								nq = 1 + iqaend - iqb
								iqaend = minint(iqaend+(*kd), iqend)
								goblas.Drot(&nq, q.Vector(iqb-1, j-1-1), func() *int { y := 1; return &y }(), q.Vector(iqb-1, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), work.GetPtr(j-1))
							}
						} else {

							for j = j1; j <= j2; j += kd1 {
								goblas.Drot(n, q.Vector(0, j-1-1), func() *int { y := 1; return &y }(), q.Vector(0, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), work.GetPtr(j-1))
							}
						}
					}

					if j2+kdn > (*n) {
						//                    adjust J2 to keep within the bounds of the matrix
						nr = nr - 1
						j2 = j2 - kdn - 1
					}

					for j = j1; j <= j2; j += kd1 {
						//                    create nonzero element a(j+kd,j-1) outside the
						//                    band and store it in WORK
						work.Set(j+(*kd)-1, work.Get(j-1)*ab.Get(kd1-1, j-1))
						ab.Set(kd1-1, j-1, d.Get(j-1)*ab.Get(kd1-1, j-1))
					}
				}
			}
		}

		if (*kd) > 0 {
			//           copy off-diagonal elements to E
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, ab.Get(1, i-1))
			}
		} else {
			//           set E to zero if original matrix was diagonal
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, zero)
			}
		}

		//        copy diagonal elements to D
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, ab.Get(0, i-1))
		}
	}
}
