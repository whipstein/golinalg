package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtgevc computes some or all of the right and/or left eigenvectors of
// a pair of real matrices (S,P), where S is a quasi-triangular matrix
// and P is upper triangular.  Matrix pairs of this type are produced by
// the generalized Schur factorization of a matrix pair (A,B):
//
//    A = Q*S*Z**T,  B = Q*P*Z**T
//
// as computed by DGGHRD + DHGEQZ.
//
// The right eigenvector x and the left eigenvector y of (S,P)
// corresponding to an eigenvalue w are defined by:
//
//    S*x = w*P*x,  (y**H)*S = w*(y**H)*P,
//
// where y**H denotes the conjugate tranpose of y.
// The eigenvalues are not input to this routine, but are computed
// directly from the diagonal blocks of S and P.
//
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of (S,P), or the products Z*X and/or Q*Y,
// where Z and Q are input matrices.
// If Q and Z are the orthogonal factors from the generalized Schur
// factorization of a matrix pair (A,B), then Z*X and Q*Y
// are the matrices of right and left eigenvectors of (A,B).
func Dtgevc(side, howmny byte, _select []bool, n *int, s *mat.Matrix, lds *int, p *mat.Matrix, ldp *int, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr, mm, m *int, work *mat.Vector, info *int) {
	var compl, compr, il2by2, ilabad, ilall, ilback, ilbbad, ilcomp, ilcplx, lsa, lsb bool
	var acoef, acoefa, anorm, ascale, bcoefa, bcoefi, bcoefr, big, bignum, bnorm, bscale, cim2a, cim2b, cimaga, cimagb, cre2a, cre2b, creala, crealb, dmin, one, safety, safmin, salfar, sbeta, scale, small, temp, temp2, temp2i, temp2r, ulp, xmax, xscale, zero float64
	var i, ibeg, ieig, iend, ihwmny, iinfo, im, iside, j, ja, jc, je, jr, jw, na, nw int

	bdiag := vf(2)
	sum := mf(2, 2, opts)
	sump := mf(2, 2, opts)
	sums := mf(2, 2, opts)

	zero = 0.0
	one = 1.0
	safety = 1.0e+2

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
		ilall = true
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
		gltest.Xerbla([]byte("DTGEVC"), -(*info))
		return
	}

	//     Count the number of eigenvectors to be computed
	if !ilall {
		im = 0
		ilcplx = false
		for j = 1; j <= (*n); j++ {
			if ilcplx {
				ilcplx = false
				goto label10
			}
			if j < (*n) {
				if s.Get(j+1-1, j-1) != zero {
					ilcplx = true
				}
			}
			if ilcplx {
				if _select[j-1] || _select[j+1-1] {
					im = im + 2
				}
			} else {
				if _select[j-1] {
					im = im + 1
				}
			}
		label10:
		}
	} else {
		im = (*n)
	}

	//     Check 2-by-2 diagonal blocks of A, B
	ilabad = false
	ilbbad = false
	for j = 1; j <= (*n)-1; j++ {
		if s.Get(j+1-1, j-1) != zero {
			if p.Get(j-1, j-1) == zero || p.Get(j+1-1, j+1-1) == zero || p.Get(j-1, j+1-1) != zero {
				ilbbad = true
			}
			if j < (*n)-1 {
				if s.Get(j+2-1, j+1-1) != zero {
					ilabad = true
				}
			}
		}
	}

	if ilabad {
		(*info) = -5
	} else if ilbbad {
		(*info) = -7
	} else if compl && (*ldvl) < (*n) || (*ldvl) < 1 {
		(*info) = -10
	} else if compr && (*ldvr) < (*n) || (*ldvr) < 1 {
		(*info) = -12
	} else if (*mm) < im {
		(*info) = -13
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTGEVC"), -(*info))
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
	//     part (i.e., excluding all elements belonging to the diagonal
	//     blocks) of A and B to check for possible overflow in the
	//     triangular solver.
	anorm = math.Abs(s.Get(0, 0))
	if (*n) > 1 {
		anorm = anorm + math.Abs(s.Get(1, 0))
	}
	bnorm = math.Abs(p.Get(0, 0))
	work.Set(0, zero)
	work.Set((*n)+1-1, zero)

	for j = 2; j <= (*n); j++ {
		temp = zero
		temp2 = zero
		if s.Get(j-1, j-1-1) == zero {
			iend = j - 1
		} else {
			iend = j - 2
		}
		for i = 1; i <= iend; i++ {
			temp = temp + math.Abs(s.Get(i-1, j-1))
			temp2 = temp2 + math.Abs(p.Get(i-1, j-1))
		}
		work.Set(j-1, temp)
		work.Set((*n)+j-1, temp2)
		for i = iend + 1; i <= minint(j+1, *n); i++ {
			temp = temp + math.Abs(s.Get(i-1, j-1))
			temp2 = temp2 + math.Abs(p.Get(i-1, j-1))
		}
		anorm = maxf64(anorm, temp)
		bnorm = maxf64(bnorm, temp2)
	}

	ascale = one / maxf64(anorm, safmin)
	bscale = one / maxf64(bnorm, safmin)

	//     Left eigenvectors
	if compl {
		ieig = 0

		//        Main loop over eigenvalues
		ilcplx = false
		for je = 1; je <= (*n); je++ {
			//           Skip this iteration if (a) HOWMNY='S' and SELECT=.FALSE., or
			//           (b) this would be the second of a complex pair.
			//           Check for complex eigenvalue, so as to be sure of which
			//           entry(-ies) of SELECT to look at.
			if ilcplx {
				ilcplx = false
				goto label220
			}
			nw = 1
			if je < (*n) {
				if s.Get(je+1-1, je-1) != zero {
					ilcplx = true
					nw = 2
				}
			}
			if ilall {
				ilcomp = true
			} else if ilcplx {
				ilcomp = _select[je-1] || _select[je+1-1]
			} else {
				ilcomp = _select[je-1]
			}
			if !ilcomp {
				goto label220
			}

			//           Decide if (a) singular pencil, (b) real eigenvalue, or
			//           (c) complex eigenvalue.
			if !ilcplx {
				if math.Abs(s.Get(je-1, je-1)) <= safmin && math.Abs(p.Get(je-1, je-1)) <= safmin {
					//                 Singular matrix pencil -- return unit eigenvector
					ieig = ieig + 1
					for jr = 1; jr <= (*n); jr++ {
						vl.Set(jr-1, ieig-1, zero)
					}
					vl.Set(ieig-1, ieig-1, one)
					goto label220
				}
			}

			//           Clear vector
			for jr = 1; jr <= nw*(*n); jr++ {
				work.Set(2*(*n)+jr-1, zero)
			}

			//           Compute coefficients in  ( a A - b B )  y = 0
			//              a  is  ACOEF
			//              b  is  BCOEFR + i*BCOEFI
			if !ilcplx {
				//              Real eigenvalue
				temp = one / maxf64(math.Abs(s.Get(je-1, je-1))*ascale, math.Abs(p.Get(je-1, je-1))*bscale, safmin)
				salfar = (temp * s.Get(je-1, je-1)) * ascale
				sbeta = (temp * p.Get(je-1, je-1)) * bscale
				acoef = sbeta * ascale
				bcoefr = salfar * bscale
				bcoefi = zero

				//              Scale to avoid underflow
				scale = one
				lsa = math.Abs(sbeta) >= safmin && math.Abs(acoef) < small
				lsb = math.Abs(salfar) >= safmin && math.Abs(bcoefr) < small
				if lsa {
					scale = (small / math.Abs(sbeta)) * minf64(anorm, big)
				}
				if lsb {
					scale = maxf64(scale, (small/math.Abs(salfar))*minf64(bnorm, big))
				}
				if lsa || lsb {
					scale = minf64(scale, one/(safmin*maxf64(one, math.Abs(acoef), math.Abs(bcoefr))))
					if lsa {
						acoef = ascale * (scale * sbeta)
					} else {
						acoef = scale * acoef
					}
					if lsb {
						bcoefr = bscale * (scale * salfar)
					} else {
						bcoefr = scale * bcoefr
					}
				}
				acoefa = math.Abs(acoef)
				bcoefa = math.Abs(bcoefr)

				//              First component is 1
				work.Set(2*(*n)+je-1, one)
				xmax = one
			} else {
				//              Complex eigenvalue
				Dlag2(s.Off(je-1, je-1), lds, p.Off(je-1, je-1), ldp, toPtrf64(safmin*safety), &acoef, &temp, &bcoefr, &temp2, &bcoefi)
				bcoefi = -bcoefi
				if bcoefi == zero {
					(*info) = je
					return
				}

				//              Scale to avoid over/underflow
				acoefa = math.Abs(acoef)
				bcoefa = math.Abs(bcoefr) + math.Abs(bcoefi)
				scale = one
				if acoefa*ulp < safmin && acoefa >= safmin {
					scale = (safmin / ulp) / acoefa
				}
				if bcoefa*ulp < safmin && bcoefa >= safmin {
					scale = maxf64(scale, (safmin/ulp)/bcoefa)
				}
				if safmin*acoefa > ascale {
					scale = ascale / (safmin * acoefa)
				}
				if safmin*bcoefa > bscale {
					scale = minf64(scale, bscale/(safmin*bcoefa))
				}
				if scale != one {
					acoef = scale * acoef
					acoefa = math.Abs(acoef)
					bcoefr = scale * bcoefr
					bcoefi = scale * bcoefi
					bcoefa = math.Abs(bcoefr) + math.Abs(bcoefi)
				}

				//              Compute first two components of eigenvector
				temp = acoef * s.Get(je+1-1, je-1)
				temp2r = acoef*s.Get(je-1, je-1) - bcoefr*p.Get(je-1, je-1)
				temp2i = -bcoefi * p.Get(je-1, je-1)
				if math.Abs(temp) > math.Abs(temp2r)+math.Abs(temp2i) {
					work.Set(2*(*n)+je-1, one)
					work.Set(3*(*n)+je-1, zero)
					work.Set(2*(*n)+je+1-1, -temp2r/temp)
					work.Set(3*(*n)+je+1-1, -temp2i/temp)
				} else {
					work.Set(2*(*n)+je+1-1, one)
					work.Set(3*(*n)+je+1-1, zero)
					temp = acoef * s.Get(je-1, je+1-1)
					work.Set(2*(*n)+je-1, (bcoefr*p.Get(je+1-1, je+1-1)-acoef*s.Get(je+1-1, je+1-1))/temp)
					work.Set(3*(*n)+je-1, bcoefi*p.Get(je+1-1, je+1-1)/temp)
				}
				xmax = maxf64(math.Abs(work.Get(2*(*n)+je-1))+math.Abs(work.Get(3*(*n)+je-1)), math.Abs(work.Get(2*(*n)+je+1-1))+math.Abs(work.Get(3*(*n)+je+1-1)))
			}

			dmin = maxf64(ulp*acoefa*anorm, ulp*bcoefa*bnorm, safmin)

			//                                           T
			//           Triangular solve of  (a A - b B)  y = 0
			//
			//                                   T
			//           (rowwise in  (a A - b B) , or columnwise in (a A - b B) )
			il2by2 = false

			for j = je + nw; j <= (*n); j++ {
				if il2by2 {
					il2by2 = false
					goto label160
				}

				na = 1
				bdiag.Set(0, p.Get(j-1, j-1))
				if j < (*n) {
					if s.Get(j+1-1, j-1) != zero {
						il2by2 = true
						bdiag.Set(1, p.Get(j+1-1, j+1-1))
						na = 2
					}
				}

				//              Check whether scaling is necessary for dot products
				xscale = one / maxf64(one, xmax)
				temp = maxf64(work.Get(j-1), work.Get((*n)+j-1), acoefa*work.Get(j-1)+bcoefa*work.Get((*n)+j-1))
				if il2by2 {
					temp = maxf64(temp, work.Get(j+1-1), work.Get((*n)+j+1-1), acoefa*work.Get(j+1-1)+bcoefa*work.Get((*n)+j+1-1))
				}
				if temp > bignum*xscale {
					for jw = 0; jw <= nw-1; jw++ {
						for jr = je; jr <= j-1; jr++ {
							work.Set((jw+2)*(*n)+jr-1, xscale*work.Get((jw+2)*(*n)+jr-1))
						}
					}
					xmax = xmax * xscale
				}

				//              Compute dot products
				//
				//                    j-1
				//              SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k)
				//                    k=je
				//
				//              To reduce the op count, this is done as
				//
				//              _        j-1                  _        j-1
				//              a*conjg( sum  S(k,j)*x(k) ) - b*conjg( sum  P(k,j)*x(k) )
				//                       k=je                          k=je
				//
				//              which may cause underflow problems if A or B are close
				//              to underflow.  (E.g., less than SMALL.)
				for jw = 1; jw <= nw; jw++ {
					for ja = 1; ja <= na; ja++ {
						sums.Set(ja-1, jw-1, zero)
						sump.Set(ja-1, jw-1, zero)

						for jr = je; jr <= j-1; jr++ {
							sums.Set(ja-1, jw-1, sums.Get(ja-1, jw-1)+s.Get(jr-1, j+ja-1-1)*work.Get((jw+1)*(*n)+jr-1))
							sump.Set(ja-1, jw-1, sump.Get(ja-1, jw-1)+p.Get(jr-1, j+ja-1-1)*work.Get((jw+1)*(*n)+jr-1))
						}
					}
				}

				for ja = 1; ja <= na; ja++ {
					if ilcplx {
						sum.Set(ja-1, 0, -acoef*sums.Get(ja-1, 0)+bcoefr*sump.Get(ja-1, 0)-bcoefi*sump.Get(ja-1, 1))
						sum.Set(ja-1, 1, -acoef*sums.Get(ja-1, 1)+bcoefr*sump.Get(ja-1, 1)+bcoefi*sump.Get(ja-1, 0))
					} else {
						sum.Set(ja-1, 0, -acoef*sums.Get(ja-1, 0)+bcoefr*sump.Get(ja-1, 0))
					}
				}

				//                                  T
				//              Solve  ( a A - b B )  y = SUM(,)
				//              with scaling and perturbation of the denominator
				Dlaln2(true, &na, &nw, &dmin, &acoef, s.Off(j-1, j-1), lds, bdiag.GetPtr(0), bdiag.GetPtr(1), sum, func() *int { y := 2; return &y }(), &bcoefr, &bcoefi, work.MatrixOff(2*(*n)+j-1, *n, opts), n, &scale, &temp, &iinfo)
				if scale < one {
					for jw = 0; jw <= nw-1; jw++ {
						for jr = je; jr <= j-1; jr++ {
							work.Set((jw+2)*(*n)+jr-1, scale*work.Get((jw+2)*(*n)+jr-1))
						}
					}
					xmax = scale * xmax
				}
				xmax = maxf64(xmax, temp)
			label160:
			}

			//           Copy eigenvector to VL, back transforming if
			//           HOWMNY='B'.
			ieig = ieig + 1
			if ilback {
				for jw = 0; jw <= nw-1; jw++ {
					goblas.Dgemv(NoTrans, n, toPtr((*n)+1-je), &one, vl.Off(0, je-1), ldvl, work.Off((jw+2)*(*n)+je-1), func() *int { y := 1; return &y }(), &zero, work.Off((jw+4)*(*n)+1-1), func() *int { y := 1; return &y }())
				}
				Dlacpy(' ', n, &nw, work.MatrixOff(4*(*n)+1-1, *n, opts), n, vl.Off(0, je-1), ldvl)
				ibeg = 1
			} else {
				Dlacpy(' ', n, &nw, work.MatrixOff(2*(*n)+1-1, *n, opts), n, vl.Off(0, ieig-1), ldvl)
				ibeg = je
			}

			//           Scale eigenvector
			xmax = zero
			if ilcplx {
				for j = ibeg; j <= (*n); j++ {
					xmax = maxf64(xmax, math.Abs(vl.Get(j-1, ieig-1))+math.Abs(vl.Get(j-1, ieig+1-1)))
				}
			} else {
				for j = ibeg; j <= (*n); j++ {
					xmax = maxf64(xmax, math.Abs(vl.Get(j-1, ieig-1)))
				}
			}

			if xmax > safmin {
				xscale = one / xmax

				for jw = 0; jw <= nw-1; jw++ {
					for jr = ibeg; jr <= (*n); jr++ {
						vl.Set(jr-1, ieig+jw-1, xscale*vl.Get(jr-1, ieig+jw-1))
					}
				}
			}
			ieig = ieig + nw - 1

		label220:
		}
	}

	//     Right eigenvectors
	if compr {
		ieig = im + 1

		//        Main loop over eigenvalues
		ilcplx = false
		for je = (*n); je >= 1; je-- {
			//           Skip this iteration if (a) HOWMNY='S' and SELECT=.FALSE., or
			//           (b) this would be the second of a complex pair.
			//           Check for complex eigenvalue, so as to be sure of which
			//           entry(-ies) of SELECT to look at -- if complex, SELECT(JE)
			//           or SELECT(JE-1).
			//           If this is a complex pair, the 2-by-2 diagonal block
			//           corresponding to the eigenvalue is in rows/columns JE-1:JE
			if ilcplx {
				ilcplx = false
				goto label500
			}
			nw = 1
			if je > 1 {
				if s.Get(je-1, je-1-1) != zero {
					ilcplx = true
					nw = 2
				}
			}
			if ilall {
				ilcomp = true
			} else if ilcplx {
				ilcomp = _select[je-1] || _select[je-1-1]
			} else {
				ilcomp = _select[je-1]
			}
			if !ilcomp {
				goto label500
			}

			//           Decide if (a) singular pencil, (b) real eigenvalue, or
			//           (c) complex eigenvalue.
			if !ilcplx {
				if math.Abs(s.Get(je-1, je-1)) <= safmin && math.Abs(p.Get(je-1, je-1)) <= safmin {
					//                 Singular matrix pencil -- unit eigenvector
					ieig = ieig - 1
					for jr = 1; jr <= (*n); jr++ {
						vr.Set(jr-1, ieig-1, zero)
					}
					vr.Set(ieig-1, ieig-1, one)
					goto label500
				}
			}

			//           Clear vector
			for jw = 0; jw <= nw-1; jw++ {
				for jr = 1; jr <= (*n); jr++ {
					work.Set((jw+2)*(*n)+jr-1, zero)
				}
			}

			//           Compute coefficients in  ( a A - b B ) x = 0
			//              a  is  ACOEF
			//              b  is  BCOEFR + i*BCOEFI
			if !ilcplx {
				//              Real eigenvalue
				temp = one / maxf64(math.Abs(s.Get(je-1, je-1))*ascale, math.Abs(p.Get(je-1, je-1))*bscale, safmin)
				salfar = (temp * s.Get(je-1, je-1)) * ascale
				sbeta = (temp * p.Get(je-1, je-1)) * bscale
				acoef = sbeta * ascale
				bcoefr = salfar * bscale
				bcoefi = zero

				//              Scale to avoid underflow
				scale = one
				lsa = math.Abs(sbeta) >= safmin && math.Abs(acoef) < small
				lsb = math.Abs(salfar) >= safmin && math.Abs(bcoefr) < small
				if lsa {
					scale = (small / math.Abs(sbeta)) * minf64(anorm, big)
				}
				if lsb {
					scale = maxf64(scale, (small/math.Abs(salfar))*minf64(bnorm, big))
				}
				if lsa || lsb {
					scale = minf64(scale, one/(safmin*maxf64(one, math.Abs(acoef), math.Abs(bcoefr))))
					if lsa {
						acoef = ascale * (scale * sbeta)
					} else {
						acoef = scale * acoef
					}
					if lsb {
						bcoefr = bscale * (scale * salfar)
					} else {
						bcoefr = scale * bcoefr
					}
				}
				acoefa = math.Abs(acoef)
				bcoefa = math.Abs(bcoefr)

				//              First component is 1
				work.Set(2*(*n)+je-1, one)
				xmax = one

				//              Compute contribution from column JE of A and B to sum
				//              (See "Further Details", above.)
				for jr = 1; jr <= je-1; jr++ {
					work.Set(2*(*n)+jr-1, bcoefr*p.Get(jr-1, je-1)-acoef*s.Get(jr-1, je-1))
				}
			} else {
				//              Complex eigenvalue
				Dlag2(s.Off(je-1-1, je-1-1), lds, p.Off(je-1-1, je-1-1), ldp, toPtrf64(safmin*safety), &acoef, &temp, &bcoefr, &temp2, &bcoefi)
				if bcoefi == zero {
					(*info) = je - 1
					return
				}

				//              Scale to avoid over/underflow
				acoefa = math.Abs(acoef)
				bcoefa = math.Abs(bcoefr) + math.Abs(bcoefi)
				scale = one
				if acoefa*ulp < safmin && acoefa >= safmin {
					scale = (safmin / ulp) / acoefa
				}
				if bcoefa*ulp < safmin && bcoefa >= safmin {
					scale = maxf64(scale, (safmin/ulp)/bcoefa)
				}
				if safmin*acoefa > ascale {
					scale = ascale / (safmin * acoefa)
				}
				if safmin*bcoefa > bscale {
					scale = minf64(scale, bscale/(safmin*bcoefa))
				}
				if scale != one {
					acoef = scale * acoef
					acoefa = math.Abs(acoef)
					bcoefr = scale * bcoefr
					bcoefi = scale * bcoefi
					bcoefa = math.Abs(bcoefr) + math.Abs(bcoefi)
				}

				//              Compute first two components of eigenvector
				//              and contribution to sums
				temp = acoef * s.Get(je-1, je-1-1)
				temp2r = acoef*s.Get(je-1, je-1) - bcoefr*p.Get(je-1, je-1)
				temp2i = -bcoefi * p.Get(je-1, je-1)
				if math.Abs(temp) >= math.Abs(temp2r)+math.Abs(temp2i) {
					work.Set(2*(*n)+je-1, one)
					work.Set(3*(*n)+je-1, zero)
					work.Set(2*(*n)+je-1-1, -temp2r/temp)
					work.Set(3*(*n)+je-1-1, -temp2i/temp)
				} else {
					work.Set(2*(*n)+je-1-1, one)
					work.Set(3*(*n)+je-1-1, zero)
					temp = acoef * s.Get(je-1-1, je-1)
					work.Set(2*(*n)+je-1, (bcoefr*p.Get(je-1-1, je-1-1)-acoef*s.Get(je-1-1, je-1-1))/temp)
					work.Set(3*(*n)+je-1, bcoefi*p.Get(je-1-1, je-1-1)/temp)
				}

				xmax = maxf64(math.Abs(work.Get(2*(*n)+je-1))+math.Abs(work.Get(3*(*n)+je-1)), math.Abs(work.Get(2*(*n)+je-1-1))+math.Abs(work.Get(3*(*n)+je-1-1)))

				//              Compute contribution from columns JE and JE-1
				//              of A and B to the sums.
				creala = acoef * work.Get(2*(*n)+je-1-1)
				cimaga = acoef * work.Get(3*(*n)+je-1-1)
				crealb = bcoefr*work.Get(2*(*n)+je-1-1) - bcoefi*work.Get(3*(*n)+je-1-1)
				cimagb = bcoefi*work.Get(2*(*n)+je-1-1) + bcoefr*work.Get(3*(*n)+je-1-1)
				cre2a = acoef * work.Get(2*(*n)+je-1)
				cim2a = acoef * work.Get(3*(*n)+je-1)
				cre2b = bcoefr*work.Get(2*(*n)+je-1) - bcoefi*work.Get(3*(*n)+je-1)
				cim2b = bcoefi*work.Get(2*(*n)+je-1) + bcoefr*work.Get(3*(*n)+je-1)
				for jr = 1; jr <= je-2; jr++ {
					work.Set(2*(*n)+jr-1, -creala*s.Get(jr-1, je-1-1)+crealb*p.Get(jr-1, je-1-1)-cre2a*s.Get(jr-1, je-1)+cre2b*p.Get(jr-1, je-1))
					work.Set(3*(*n)+jr-1, -cimaga*s.Get(jr-1, je-1-1)+cimagb*p.Get(jr-1, je-1-1)-cim2a*s.Get(jr-1, je-1)+cim2b*p.Get(jr-1, je-1))
				}
			}

			dmin = maxf64(ulp*acoefa*anorm, ulp*bcoefa*bnorm, safmin)

			//           Columnwise triangular solve of  (a A - b B)  x = 0
			il2by2 = false
			for j = je - nw; j >= 1; j-- {
				//              If a 2-by-2 block, is in position j-1:j, wait until
				//              next iteration to process it (when it will be j:j+1)
				if !il2by2 && j > 1 {
					if s.Get(j-1, j-1-1) != zero {
						il2by2 = true
						goto label370
					}
				}
				bdiag.Set(0, p.Get(j-1, j-1))
				if il2by2 {
					na = 2
					bdiag.Set(1, p.Get(j+1-1, j+1-1))
				} else {
					na = 1
				}

				//              Compute x(j) (and x(j+1), if 2-by-2 block)
				Dlaln2(false, &na, &nw, &dmin, &acoef, s.Off(j-1, j-1), lds, bdiag.GetPtr(0), bdiag.GetPtr(1), work.MatrixOff(2*(*n)+j-1, *n, opts), n, &bcoefr, &bcoefi, sum, func() *int { y := 2; return &y }(), &scale, &temp, &iinfo)
				if scale < one {

					for jw = 0; jw <= nw-1; jw++ {
						for jr = 1; jr <= je; jr++ {
							work.Set((jw+2)*(*n)+jr-1, scale*work.Get((jw+2)*(*n)+jr-1))
						}
					}
				}
				xmax = maxf64(scale*xmax, temp)

				for jw = 1; jw <= nw; jw++ {
					for ja = 1; ja <= na; ja++ {
						work.Set((jw+1)*(*n)+j+ja-1-1, sum.Get(ja-1, jw-1))
					}
				}

				//              w = w + x(j)*(a S(*,j) - b P(*,j) ) with scaling
				if j > 1 {
					//                 Check whether scaling is necessary for sum.
					xscale = one / maxf64(one, xmax)
					temp = acoefa*work.Get(j-1) + bcoefa*work.Get((*n)+j-1)
					if il2by2 {
						temp = maxf64(temp, acoefa*work.Get(j+1-1)+bcoefa*work.Get((*n)+j+1-1))
					}
					temp = maxf64(temp, acoefa, bcoefa)
					if temp > bignum*xscale {

						for jw = 0; jw <= nw-1; jw++ {
							for jr = 1; jr <= je; jr++ {
								work.Set((jw+2)*(*n)+jr-1, xscale*work.Get((jw+2)*(*n)+jr-1))
							}
						}
						xmax = xmax * xscale
					}

					//                 Compute the contributions of the off-diagonals of
					//                 column j (and j+1, if 2-by-2 block) of A and B to the
					//                 sums.
					for ja = 1; ja <= na; ja++ {
						if ilcplx {
							creala = acoef * work.Get(2*(*n)+j+ja-1-1)
							cimaga = acoef * work.Get(3*(*n)+j+ja-1-1)
							crealb = bcoefr*work.Get(2*(*n)+j+ja-1-1) - bcoefi*work.Get(3*(*n)+j+ja-1-1)
							cimagb = bcoefi*work.Get(2*(*n)+j+ja-1-1) + bcoefr*work.Get(3*(*n)+j+ja-1-1)
							for jr = 1; jr <= j-1; jr++ {
								work.Set(2*(*n)+jr-1, work.Get(2*(*n)+jr-1)-creala*s.Get(jr-1, j+ja-1-1)+crealb*p.Get(jr-1, j+ja-1-1))
								work.Set(3*(*n)+jr-1, work.Get(3*(*n)+jr-1)-cimaga*s.Get(jr-1, j+ja-1-1)+cimagb*p.Get(jr-1, j+ja-1-1))
							}
						} else {
							creala = acoef * work.Get(2*(*n)+j+ja-1-1)
							crealb = bcoefr * work.Get(2*(*n)+j+ja-1-1)
							for jr = 1; jr <= j-1; jr++ {
								work.Set(2*(*n)+jr-1, work.Get(2*(*n)+jr-1)-creala*s.Get(jr-1, j+ja-1-1)+crealb*p.Get(jr-1, j+ja-1-1))
							}
						}
					}
				}

				il2by2 = false
			label370:
			}

			//           Copy eigenvector to VR, back transforming if
			//           HOWMNY='B'.
			ieig = ieig - nw
			if ilback {

				for jw = 0; jw <= nw-1; jw++ {
					for jr = 1; jr <= (*n); jr++ {
						work.Set((jw+4)*(*n)+jr-1, work.Get((jw+2)*(*n)+1-1)*vr.Get(jr-1, 0))
					}

					//                 A series of compiler directives to defeat
					//                 vectorization for the next loop
					for jc = 2; jc <= je; jc++ {
						for jr = 1; jr <= (*n); jr++ {
							work.Set((jw+4)*(*n)+jr-1, work.Get((jw+4)*(*n)+jr-1)+work.Get((jw+2)*(*n)+jc-1)*vr.Get(jr-1, jc-1))
						}
					}
				}

				for jw = 0; jw <= nw-1; jw++ {
					for jr = 1; jr <= (*n); jr++ {
						vr.Set(jr-1, ieig+jw-1, work.Get((jw+4)*(*n)+jr-1))
					}
				}

				iend = (*n)
			} else {
				for jw = 0; jw <= nw-1; jw++ {
					for jr = 1; jr <= (*n); jr++ {
						vr.Set(jr-1, ieig+jw-1, work.Get((jw+2)*(*n)+jr-1))
					}
				}

				iend = je
			}

			//           Scale eigenvector
			xmax = zero
			if ilcplx {
				for j = 1; j <= iend; j++ {
					xmax = maxf64(xmax, math.Abs(vr.Get(j-1, ieig-1))+math.Abs(vr.Get(j-1, ieig+1-1)))
				}
			} else {
				for j = 1; j <= iend; j++ {
					xmax = maxf64(xmax, math.Abs(vr.Get(j-1, ieig-1)))
				}
			}

			if xmax > safmin {
				xscale = one / xmax
				for jw = 0; jw <= nw-1; jw++ {
					for jr = 1; jr <= iend; jr++ {
						vr.Set(jr-1, ieig+jw-1, xscale*vr.Get(jr-1, ieig+jw-1))
					}
				}
			}
		label500:
		}
	}
}
