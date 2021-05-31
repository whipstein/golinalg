package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgevc computes some or all of the right and/or left eigenvectors of
// a pair of complex matrices (S,P), where S and P are upper triangular.
// Matrix pairs of this _type are produced by the generalized Schur
// factorization of a complex matrix pair (A,B):
//
//    A = Q*S*Z**H,  B = Q*P*Z**H
//
// as computed by ZGGHRD + ZHGEQZ.
//
// The right eigenvector x and the left eigenvector y of (S,P)
// corresponding to an eigenvalue w are defined by:
//
//    S*x = w*P*x,  (y**H)*S = w*(y**H)*P,
//
// where y**H denotes the conjugate tranpose of y.
// The eigenvalues are not input to this routine, but are computed
// directly from the diagonal elements of S and P.
//
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of (S,P), or the products Z*X and/or Q*Y,
// where Z and Q are input matrices.
// If Q and Z are the unitary factors from the generalized Schur
// factorization of a matrix pair (A,B), then Z*X and Q*Y
// are the matrices of right and left eigenvectors of (A,B).
func Ztgevc(side, howmny byte, _select []bool, n *int, s *mat.CMatrix, lds *int, p *mat.CMatrix, ldp *int, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr, mm, m *int, work *mat.CVector, rwork *mat.Vector, info *int) {
	var compl, compr, ilall, ilback, ilbbad, ilcomp, lsa, lsb bool
	var bcoeff, ca, cb, cone, czero, d, salpha, sum, suma, sumb complex128
	var acoefa, acoeff, anorm, ascale, bcoefa, big, bignum, bnorm, bscale, dmin, one, safmin, sbeta, scale, small, temp, ulp, xmax, zero float64
	var i, ibeg, ieig, iend, ihwmny, im, iside, isrc, j, je, jr int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Decode and Test the input parameters
	if howmny == 'A' {
		ihwmny = 1
		ilall = true
		ilback = false
	} else if howmny == 'S' {
		ihwmny = 2
		ilall = false
		ilback = false
	} else if howmny == 'B' {
		ihwmny = 3
		ilall = true
		ilback = true
	} else {
		ihwmny = -1
	}

	if side == 'R' {
		iside = 1
		compl = false
		compr = true
	} else if side == 'L' {
		iside = 2
		compl = true
		compr = false
	} else if side == 'B' {
		iside = 3
		compl = true
		compr = true
	} else {
		iside = -1
	}

	(*info) = 0
	if iside < 0 {
		(*info) = -1
	} else if ihwmny < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lds) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldp) < maxint(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGEVC"), -(*info))
		return
	}

	//     Count the number of eigenvectors
	if !ilall {
		im = 0
		for j = 1; j <= (*n); j++ {
			if _select[j-1] {
				im = im + 1
			}
		}
	} else {
		im = (*n)
	}

	//     Check diagonal of B
	ilbbad = false
	for j = 1; j <= (*n); j++ {
		if p.GetIm(j-1, j-1) != zero {
			ilbbad = true
		}
	}

	if ilbbad {
		(*info) = -7
	} else if compl && (*ldvl) < (*n) || (*ldvl) < 1 {
		(*info) = -10
	} else if compr && (*ldvr) < (*n) || (*ldvr) < 1 {
		(*info) = -12
	} else if (*mm) < im {
		(*info) = -13
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGEVC"), -(*info))
		return
	}

	//     Quick return if possible
	(*m) = im
	if (*n) == 0 {
		return
	}

	//     Machine Constants
	safmin = Dlamch(SafeMinimum)
	big = one / safmin
	Dlabad(&safmin, &big)
	ulp = Dlamch(Epsilon) * Dlamch(Base)
	small = safmin * float64(*n) / ulp
	big = one / small
	bignum = one / (safmin * float64(*n))

	//     Compute the 1-norm of each column of the strictly upper triangular
	//     part of A and B to check for possible overflow in the triangular
	//     solver.
	anorm = abs1(s.Get(0, 0))
	bnorm = abs1(p.Get(0, 0))
	rwork.Set(0, zero)
	rwork.Set((*n)+1-1, zero)
	for j = 2; j <= (*n); j++ {
		rwork.Set(j-1, zero)
		rwork.Set((*n)+j-1, zero)
		for i = 1; i <= j-1; i++ {
			rwork.Set(j-1, rwork.Get(j-1)+abs1(s.Get(i-1, j-1)))
			rwork.Set((*n)+j-1, rwork.Get((*n)+j-1)+abs1(p.Get(i-1, j-1)))
		}
		anorm = maxf64(anorm, rwork.Get(j-1)+abs1(s.Get(j-1, j-1)))
		bnorm = maxf64(bnorm, rwork.Get((*n)+j-1)+abs1(p.Get(j-1, j-1)))
	}

	ascale = one / maxf64(anorm, safmin)
	bscale = one / maxf64(bnorm, safmin)

	//     Left eigenvectors
	if compl {
		ieig = 0

		//        Main loop over eigenvalues
		for je = 1; je <= (*n); je++ {
			if ilall {
				ilcomp = true
			} else {
				ilcomp = _select[je-1]
			}
			if ilcomp {
				ieig = ieig + 1

				if abs1(s.Get(je-1, je-1)) <= safmin && math.Abs(p.GetRe(je-1, je-1)) <= safmin {
					//                 Singular matrix pencil -- return unit eigenvector
					for jr = 1; jr <= (*n); jr++ {
						vl.Set(jr-1, ieig-1, czero)
					}
					vl.Set(ieig-1, ieig-1, cone)
					goto label140
				}

				//              Non-singular eigenvalue:
				//              Compute coefficients  a  and  b  in
				//                   H
				//                 y  ( a A - b B ) = 0
				temp = one / maxf64(abs1(s.Get(je-1, je-1))*ascale, math.Abs(p.GetRe(je-1, je-1))*bscale, safmin)
				salpha = (complex(temp, 0) * s.Get(je-1, je-1)) * complex(ascale, 0)
				sbeta = (temp * p.GetRe(je-1, je-1)) * bscale
				acoeff = sbeta * ascale
				bcoeff = salpha * complex(bscale, 0)
				//
				//              Scale to avoid underflow
				//
				lsa = math.Abs(sbeta) >= safmin && math.Abs(acoeff) < small
				lsb = abs1(salpha) >= safmin && abs1(bcoeff) < small
				//
				scale = one
				if lsa {
					scale = (small / math.Abs(sbeta)) * minf64(anorm, big)
				}
				if lsb {
					scale = maxf64(scale, (small/abs1(salpha))*minf64(bnorm, big))
				}
				if lsa || lsb {
					scale = minf64(scale, one/(safmin*maxf64(one, math.Abs(acoeff), abs1(bcoeff))))
					if lsa {
						acoeff = ascale * (scale * sbeta)
					} else {
						acoeff = scale * acoeff
					}
					if lsb {
						bcoeff = complex(bscale, 0) * (complex(scale, 0) * salpha)
					} else {
						bcoeff = complex(scale, 0) * bcoeff
					}
				}
				//
				acoefa = math.Abs(acoeff)
				bcoefa = abs1(bcoeff)
				xmax = one
				for jr = 1; jr <= (*n); jr++ {
					work.Set(jr-1, czero)
				}
				work.Set(je-1, cone)
				dmin = maxf64(ulp*acoefa*anorm, ulp*bcoefa*bnorm, safmin)

				//                                              H
				//              Triangular solve of  (a A - b B)  y = 0
				//
				//                                      H
				//              (rowwise in  (a A - b B) , or columnwise in a A - b B)
				for j = je + 1; j <= (*n); j++ {
					//                 Compute
					//                       j-1
					//                 SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k)
					//                       k=je
					//                 (Scale if necessary)
					temp = one / xmax
					if acoefa*rwork.Get(j-1)+bcoefa*rwork.Get((*n)+j-1) > bignum*temp {
						for jr = je; jr <= j-1; jr++ {
							work.Set(jr-1, complex(temp, 0)*work.Get(jr-1))
						}
						xmax = one
					}
					suma = czero
					sumb = czero

					for jr = je; jr <= j-1; jr++ {
						suma = suma + s.GetConj(jr-1, j-1)*work.Get(jr-1)
						sumb = sumb + p.GetConj(jr-1, j-1)*work.Get(jr-1)
					}
					sum = complex(acoeff, 0)*suma - cmplx.Conj(bcoeff)*sumb

					//                 Form x(j) = - SUM / conjg( a*S(j,j) - b*P(j,j) )
					//
					//                 with scaling and perturbation of the denominator
					d = cmplx.Conj(complex(acoeff, 0)*s.Get(j-1, j-1) - bcoeff*p.Get(j-1, j-1))
					if abs1(d) <= dmin {
						d = complex(dmin, 0)
					}

					if abs1(d) < one {
						if abs1(sum) >= bignum*abs1(d) {
							temp = one / abs1(sum)
							for jr = je; jr <= j-1; jr++ {
								work.Set(jr-1, complex(temp, 0)*work.Get(jr-1))
							}
							xmax = temp * xmax
							sum = complex(temp, 0) * sum
						}
					}
					work.Set(j-1, Zladiv(toPtrc128(-sum), &d))
					xmax = maxf64(xmax, abs1(work.Get(j-1)))
				}

				//              Back transform eigenvector if HOWMNY='B'.
				if ilback {
					goblas.Zgemv(NoTrans, n, toPtr((*n)+1-je), &cone, vl.Off(0, je-1), ldvl, work.Off(je-1), func() *int { y := 1; return &y }(), &czero, work.Off((*n)+1-1), func() *int { y := 1; return &y }())
					isrc = 2
					ibeg = 1
				} else {
					isrc = 1
					ibeg = je
				}

				//              Copy and scale eigenvector into column of VL
				xmax = zero
				for jr = ibeg; jr <= (*n); jr++ {
					xmax = maxf64(xmax, abs1(work.Get((isrc-1)*(*n)+jr-1)))
				}

				if xmax > safmin {
					temp = one / xmax
					for jr = ibeg; jr <= (*n); jr++ {
						vl.Set(jr-1, ieig-1, complex(temp, 0)*work.Get((isrc-1)*(*n)+jr-1))
					}
				} else {
					ibeg = (*n) + 1
				}

				for jr = 1; jr <= ibeg-1; jr++ {
					vl.Set(jr-1, ieig-1, czero)
				}

			}
		label140:
		}
	}

	//     Right eigenvectors
	if compr {
		ieig = im + 1

		//        Main loop over eigenvalues
		for je = (*n); je >= 1; je -= 1 {
			if ilall {
				ilcomp = true
			} else {
				ilcomp = _select[je-1]
			}
			if ilcomp {
				ieig = ieig - 1

				if abs1(s.Get(je-1, je-1)) <= safmin && math.Abs(p.GetRe(je-1, je-1)) <= safmin {
					//                 Singular matrix pencil -- return unit eigenvector
					for jr = 1; jr <= (*n); jr++ {
						vr.Set(jr-1, ieig-1, czero)
					}
					vr.Set(ieig-1, ieig-1, cone)
					goto label250
				}

				//              Non-singular eigenvalue:
				//              Compute coefficients  a  and  b  in
				//
				//              ( a A - b B ) x  = 0
				temp = one / maxf64(abs1(s.Get(je-1, je-1))*ascale, math.Abs(p.GetRe(je-1, je-1))*bscale, safmin)
				salpha = (complex(temp, 0) * s.Get(je-1, je-1)) * complex(ascale, 0)
				sbeta = (temp * p.GetRe(je-1, je-1)) * bscale
				acoeff = sbeta * ascale
				bcoeff = salpha * complex(bscale, 0)

				//              Scale to avoid underflow
				lsa = math.Abs(sbeta) >= safmin && math.Abs(acoeff) < small
				lsb = abs1(salpha) >= safmin && abs1(bcoeff) < small

				scale = one
				if lsa {
					scale = (small / math.Abs(sbeta)) * minf64(anorm, big)
				}
				if lsb {
					scale = maxf64(scale, (small/abs1(salpha))*minf64(bnorm, big))
				}
				if lsa || lsb {
					scale = minf64(scale, one/(safmin*maxf64(one, math.Abs(acoeff), abs1(bcoeff))))
					if lsa {
						acoeff = ascale * (scale * sbeta)
					} else {
						acoeff = scale * acoeff
					}
					if lsb {
						bcoeff = complex(bscale, 0) * (complex(scale, 0) * salpha)
					} else {
						bcoeff = complex(scale, 0) * bcoeff
					}
				}

				acoefa = math.Abs(acoeff)
				bcoefa = abs1(bcoeff)
				xmax = one
				for jr = 1; jr <= (*n); jr++ {
					work.Set(jr-1, czero)
				}
				work.Set(je-1, cone)
				dmin = maxf64(ulp*acoefa*anorm, ulp*bcoefa*bnorm, safmin)

				//              Triangular solve of  (a A - b B) x = 0  (columnwise)
				//
				//              WORK(1:j-1) contains sums w,
				//              WORK(j+1:JE) contains x
				for jr = 1; jr <= je-1; jr++ {
					work.Set(jr-1, complex(acoeff, 0)*s.Get(jr-1, je-1)-bcoeff*p.Get(jr-1, je-1))
				}
				work.Set(je-1, cone)

				for j = je - 1; j >= 1; j -= 1 {
					//                 Form x(j) := - w(j) / d
					//                 with scaling and perturbation of the denominator
					d = complex(acoeff, 0)*s.Get(j-1, j-1) - bcoeff*p.Get(j-1, j-1)
					if abs1(d) <= dmin {
						d = complex(dmin, 0)
					}

					if abs1(d) < one {
						if abs1(work.Get(j-1)) >= bignum*abs1(d) {
							temp = one / abs1(work.Get(j-1))
							for jr = 1; jr <= je; jr++ {
								work.Set(jr-1, complex(temp, 0)*work.Get(jr-1))
							}
						}
					}

					work.Set(j-1, Zladiv(toPtrc128(-work.Get(j-1)), &d))

					if j > 1 {
						//                    w = w + x(j)*(a S(*,j) - b P(*,j) ) with scaling
						if abs1(work.Get(j-1)) > one {
							temp = one / abs1(work.Get(j-1))
							if acoefa*rwork.Get(j-1)+bcoefa*rwork.Get((*n)+j-1) >= bignum*temp {
								for jr = 1; jr <= je; jr++ {
									work.Set(jr-1, complex(temp, 0)*work.Get(jr-1))
								}
							}
						}

						ca = complex(acoeff, 0) * work.Get(j-1)
						cb = bcoeff * work.Get(j-1)
						for jr = 1; jr <= j-1; jr++ {
							work.Set(jr-1, work.Get(jr-1)+ca*s.Get(jr-1, j-1)-cb*p.Get(jr-1, j-1))
						}
					}
				}

				//              Back transform eigenvector if HOWMNY='B'.
				if ilback {
					goblas.Zgemv(NoTrans, n, &je, &cone, vr, ldvr, work, func() *int { y := 1; return &y }(), &czero, work.Off((*n)+1-1), func() *int { y := 1; return &y }())
					isrc = 2
					iend = (*n)
				} else {
					isrc = 1
					iend = je
				}

				//              Copy and scale eigenvector into column of VR
				xmax = zero
				for jr = 1; jr <= iend; jr++ {
					xmax = maxf64(xmax, abs1(work.Get((isrc-1)*(*n)+jr-1)))
				}

				if xmax > safmin {
					temp = one / xmax
					for jr = 1; jr <= iend; jr++ {
						vr.Set(jr-1, ieig-1, complex(temp, 0)*work.Get((isrc-1)*(*n)+jr-1))
					}
				} else {
					iend = 0
				}

				for jr = iend + 1; jr <= (*n); jr++ {
					vr.Set(jr-1, ieig-1, czero)
				}

			}
		label250:
		}
	}
}
