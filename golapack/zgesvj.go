package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgesvj computes the singular value decomposition (SVD) of a complex
// M-by-N matrix A, where M >= N. The SVD of A is written as
//                                    [++]   [xx]   [x0]   [xx]
//              A = U * SIGMA * V^*,  [++] = [xx] * [ox] * [xx]
//                                    [++]   [xx]
// where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
// matrix, and V is an N-by-N unitary matrix. The diagonal elements
// of SIGMA are the singular values of A. The columns of U and V are the
// left and the right singular vectors of A, respectively.
func Zgesvj(joba, jobu, jobv byte, m, n *int, a *mat.CMatrix, lda *int, sva *mat.Vector, mv *int, v *mat.CMatrix, ldv *int, cwork *mat.CVector, lwork *int, rwork *mat.Vector, lrwork, info *int) {
	var applv, goscale, lower, lquery, lsvec, noscale, rotok, rsvec, uctol, upper bool
	var aapq, cone, czero, ompq complex128
	var aapp, aapp0, aapq1, aaqq, apoaq, aqoap, big, bigtheta, cs, ctol, epsln, half, mxaapq, mxsinj, one, rootbig, rooteps, rootsfmin, roottol, sfmin, skl, small, sn, t, temp1, theta, thsign, tol, zero float64
	var blskip, emptsw, i, ibr, ierr, igl, ijblsk, ir1, iswrot, jbc, jgl, kbl, lkahead, mvl, n2, n34, n4, nbl, notrot, nsweep, p, pskipped, q, rowskip, swband int

	zero = 0.0
	half = 0.5
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	nsweep = 30

	//     Test the input arguments
	lsvec = jobu == 'U' || jobu == 'F'
	uctol = jobu == 'C'
	rsvec = jobv == 'V' || jobv == 'J'
	applv = jobv == 'A'
	upper = joba == 'U'
	lower = joba == 'L'

	lquery = ((*lwork) == -1) || ((*lrwork) == -1)
	if !(upper || lower || joba == 'G') {
		(*info) = -1
	} else if !(lsvec || uctol || jobu == 'N') {
		(*info) = -2
	} else if !(rsvec || applv || jobv == 'N') {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if ((*n) < 0) || ((*n) > (*m)) {
		(*info) = -5
	} else if (*lda) < (*m) {
		(*info) = -7
	} else if (*mv) < 0 {
		(*info) = -9
	} else if (rsvec && ((*ldv) < (*n))) || (applv && ((*ldv) < (*mv))) {
		(*info) = -11
	} else if uctol && (rwork.Get(0) <= one) {
		(*info) = -12
	} else if ((*lwork) < ((*m) + (*n))) && (!lquery) {
		(*info) = -13
	} else if ((*lrwork) < maxint(*n, 6)) && (!lquery) {
		(*info) = -15
	} else {
		(*info) = 0
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGESVJ"), -(*info))
		return
	} else if lquery {
		cwork.SetRe(0, float64((*m)+(*n)))
		rwork.Set(0, float64(maxint(*n, 6)))
		return
	}

	// #:) Quick return for void matrix
	if ((*m) == 0) || ((*n) == 0) {
		return
	}

	//     Set numerical parameters
	//     The stopping criterion for Jacobi rotations is
	//
	//     max_{i<>j}|A(:,i)^* * A(:,j)| / (||A(:,i)||*||A(:,j)||) < CTOL*EPS
	//
	//     where EPS is the round-off and CTOL is defined as follows:
	if uctol {
		//        ... user controlled
		ctol = rwork.Get(0)
	} else {
		//        ... default
		if lsvec || rsvec || applv {
			ctol = math.Sqrt(float64(*m))
		} else {
			ctol = float64(*m)
		}
	}
	//     ... and the machine dependent parameters are
	//[!]  (Make sure that SLAMCH() works properly on the target machine.)
	epsln = Dlamch(Epsilon)
	rooteps = math.Sqrt(epsln)
	sfmin = Dlamch(SafeMinimum)
	rootsfmin = math.Sqrt(sfmin)
	small = sfmin / epsln
	big = Dlamch(Overflow)
	//     BIG         = ONE    / SFMIN
	rootbig = one / rootsfmin
	//      LARGE = BIG / SQRT( DBLE( M*N ) )
	bigtheta = one / rooteps
	//
	tol = ctol * epsln
	roottol = math.Sqrt(tol)
	//
	if float64(*m)*epsln >= one {
		(*info) = -4
		gltest.Xerbla([]byte("ZGESVJ"), -(*info))
		return
	}

	//     Initialize the right singular vector matrix.
	if rsvec {
		mvl = (*n)
		Zlaset('A', &mvl, n, &czero, &cone, v, ldv)
	} else if applv {
		mvl = (*mv)
	}
	rsvec = rsvec || applv

	//     Initialize SVA( 1:N ) = ( ||A e_i||_2, i = 1:N )
	//(!)  If necessary, scale A to protect the largest singular value
	//     from overflow. It is possible that saving the largest singular
	//     value destroys the information about the small ones.
	//     This initial scaling is almost minimal in the sense that the
	//     goal is to make sure that no column norm overflows, and that
	//     SQRT(N)*max_i SVA(i) does not overflow. If INFinite entries
	//     in A are detected, the procedure returns with INFO=-6.
	skl = one / math.Sqrt(float64(*m)*float64(*n))
	noscale = true
	goscale = true

	if lower {
		//        the input matrix is M-by-N lower triangular (trapezoidal)
		for p = 1; p <= (*n); p++ {
			aapp = zero
			aaqq = one
			Zlassq(toPtr((*m)-p+1), a.CVector(p-1, p-1), func() *int { y := 1; return &y }(), &aapp, &aaqq)
			if aapp > big {
				(*info) = -6
				gltest.Xerbla([]byte("ZGESVJ"), -(*info))
				return
			}
			aaqq = math.Sqrt(aaqq)
			if (aapp < (big / aaqq)) && noscale {
				sva.Set(p-1, aapp*aaqq)
			} else {
				noscale = false
				sva.Set(p-1, aapp*(aaqq*skl))
				if goscale {
					goscale = false
					for q = 1; q <= p-1; q++ {
						sva.Set(q-1, sva.Get(q-1)*skl)
					}
				}
			}
		}
	} else if upper {
		//        the input matrix is M-by-N upper triangular (trapezoidal)
		for p = 1; p <= (*n); p++ {
			aapp = zero
			aaqq = one
			Zlassq(&p, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &aapp, &aaqq)
			if aapp > big {
				(*info) = -6
				gltest.Xerbla([]byte("ZGESVJ"), -(*info))
				return
			}
			aaqq = math.Sqrt(aaqq)
			if (aapp < (big / aaqq)) && noscale {
				sva.Set(p-1, aapp*aaqq)
			} else {
				noscale = false
				sva.Set(p-1, aapp*(aaqq*skl))
				if goscale {
					goscale = false
					for q = 1; q <= p-1; q++ {
						sva.Set(q-1, sva.Get(q-1)*skl)
					}
				}
			}
		}
	} else {
		//        the input matrix is M-by-N general dense
		for p = 1; p <= (*n); p++ {
			aapp = zero
			aaqq = one
			Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &aapp, &aaqq)
			if aapp > big {
				(*info) = -6
				gltest.Xerbla([]byte("ZGESVJ"), -(*info))
				return
			}
			aaqq = math.Sqrt(aaqq)
			if (aapp < (big / aaqq)) && noscale {
				sva.Set(p-1, aapp*aaqq)
			} else {
				noscale = false
				sva.Set(p-1, aapp*(aaqq*skl))
				if goscale {
					goscale = false
					for q = 1; q <= p-1; q++ {
						sva.Set(q-1, sva.Get(q-1)*skl)
					}
				}
			}
		}
	}

	if noscale {
		skl = one
	}

	//     Move the smaller part of the spectrum from the underflow threshold
	//(!)  Start by determining the position of the nonzero entries of the
	//     array SVA() relative to ( SFMIN, BIG ).
	aapp = zero
	aaqq = big
	for p = 1; p <= (*n); p++ {
		if sva.Get(p-1) != zero {
			aaqq = minf64(aaqq, sva.Get(p-1))
		}
		aapp = maxf64(aapp, sva.Get(p-1))
	}

	// #:) Quick return for zero matrix
	if aapp == zero {
		if lsvec {
			Zlaset('G', m, n, &czero, &cone, a, lda)
		}
		rwork.Set(0, one)
		rwork.Set(1, zero)
		rwork.Set(2, zero)
		rwork.Set(3, zero)
		rwork.Set(4, zero)
		rwork.Set(5, zero)
		return
	}

	// #:) Quick return for one-column matrix
	if (*n) == 1 {
		if lsvec {
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), sva.GetPtr(0), &skl, m, func() *int { y := 1; return &y }(), a, lda, &ierr)
		}
		rwork.Set(0, one/skl)
		if sva.Get(0) >= sfmin {
			rwork.Set(1, one)
		} else {
			rwork.Set(1, zero)
		}
		rwork.Set(2, zero)
		rwork.Set(3, zero)
		rwork.Set(4, zero)
		rwork.Set(5, zero)
		return
	}

	//     Protect small singular values from underflow, and try to
	//     avoid underflows/overflows in computing Jacobi rotations.
	sn = math.Sqrt(sfmin / epsln)
	temp1 = math.Sqrt(big / float64(*n))
	if (aapp <= sn) || (aaqq >= temp1) || ((sn <= aaqq) && (aapp <= temp1)) {
		temp1 = minf64(big, temp1/aapp)
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else if (aaqq <= sn) && (aapp <= temp1) {
		temp1 = minf64(sn/aaqq, big/(aapp*math.Sqrt(float64(*n))))
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else if (aaqq >= sn) && (aapp >= temp1) {
		temp1 = maxf64(sn/aaqq, temp1/aapp)
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else if (aaqq <= sn) && (aapp >= temp1) {
		temp1 = minf64(sn/aaqq, big/(math.Sqrt(float64(*n))*aapp))
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else {
		temp1 = one
	}

	//     Scale, if necessary
	if temp1 != one {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &temp1, n, func() *int { y := 1; return &y }(), sva.Matrix(*n, opts), n, &ierr)
	}
	skl = temp1 * skl
	if skl != one {
		Zlascl(joba, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &skl, m, n, a, lda, &ierr)
		skl = one / skl
	}

	//     Row-cyclic Jacobi SVD algorithm with column pivoting
	emptsw = ((*n) * ((*n) - 1)) / 2
	notrot = 0
	for q = 1; q <= (*n); q++ {
		cwork.Set(q-1, cone)
	}

	swband = 3
	//[TP] SWBAND is a tuning parameter [TP]. It is meaningful and effective
	//     if ZGESVJ is used as a computational routine in the preconditioned
	//     Jacobi SVD algorithm ZGEJSV. For sweeps i=1:SWBAND the procedure
	//     works on pivots inside a band-like region around the diagonal.
	//     The boundaries are determined dynamically, based on the number of
	//     pivots above a threshold.

	kbl = minint(int(8), *n)
	//[TP] KBL is a tuning parameter that defines the tile size in the
	//     tiling of the p-q loops of pivot pairs. In general, an optimal
	//     value of KBL depends on the matrix dimensions and on the
	//     parameters of the computer's memory.

	nbl = (*n) / kbl
	if (nbl * kbl) != (*n) {
		nbl = nbl + 1
	}

	blskip = powint(kbl, 2)
	//[TP] BLKSKIP is a tuning parameter that depends on SWBAND and KBL.

	rowskip = minint(int(5), kbl)
	//[TP] ROWSKIP is a tuning parameter.

	lkahead = 1
	//[TP] LKAHEAD is a tuning parameter.

	//     Quasi block transformations, using the lower (upper) triangular
	//     structure of the input matrix. The quasi-block-cycling usually
	//     invokes cubic convergence. Big part of this cycle is done inside
	//     canonical subspaces of dimensions less than M.
	if (lower || upper) && ((*n) > maxint(64, 4*kbl)) {
		//[TP] The number of partition levels and the actual partition are
		//     tuning parameters.
		n4 = (*n) / 4
		n2 = (*n) / 2
		n34 = 3 * n4
		if applv {
			q = 0
		} else {
			q = 1
		}

		if lower {
			//     This works very well on lower triangular matrices, in particular
			//     in the framework of the preconditioned Jacobi SVD (xGEJSV).
			//     The idea is simple:
			//     [+ 0 0 0]   Note that Jacobi transformations of [0 0]
			//     [+ + 0 0]                                       [0 0]
			//     [+ + x 0]   actually work on [x 0]              [x 0]
			//     [+ + x x]                    [x x].             [x x]
			Zgsvj0(jobv, toPtr((*m)-n34), toPtr((*n)-n34), a.Off(n34+1-1, n34+1-1), lda, cwork.Off(n34+1-1), sva.Off(n34+1-1), &mvl, v.Off(n34*q+1-1, n34+1-1), ldv, &epsln, &sfmin, &tol, func() *int { y := 2; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
			Zgsvj0(jobv, toPtr((*m)-n2), toPtr(n34-n2), a.Off(n2+1-1, n2+1-1), lda, cwork.Off(n2+1-1), sva.Off(n2+1-1), &mvl, v.Off(n2*q+1-1, n2+1-1), ldv, &epsln, &sfmin, &tol, func() *int { y := 2; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
			Zgsvj1(jobv, toPtr((*m)-n2), toPtr((*n)-n2), &n4, a.Off(n2+1-1, n2+1-1), lda, cwork.Off(n2+1-1), sva.Off(n2+1-1), &mvl, v.Off(n2*q+1-1, n2+1-1), ldv, &epsln, &sfmin, &tol, func() *int { y := 1; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
			Zgsvj0(jobv, toPtr((*m)-n4), toPtr(n2-n4), a.Off(n4+1-1, n4+1-1), lda, cwork.Off(n4+1-1), sva.Off(n4+1-1), &mvl, v.Off(n4*q+1-1, n4+1-1), ldv, &epsln, &sfmin, &tol, func() *int { y := 1; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

			Zgsvj0(jobv, m, &n4, a, lda, cwork, sva, &mvl, v, ldv, &epsln, &sfmin, &tol, func() *int { y := 1; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

			Zgsvj1(jobv, m, &n2, &n4, a, lda, cwork, sva, &mvl, v, ldv, &epsln, &sfmin, &tol, func() *int { y := 1; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

		} else if upper {
			Zgsvj0(jobv, &n4, &n4, a, lda, cwork, sva, &mvl, v, ldv, &epsln, &sfmin, &tol, func() *int { y := 2; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

			Zgsvj0(jobv, &n2, &n4, a.Off(0, n4+1-1), lda, cwork.Off(n4+1-1), sva.Off(n4+1-1), &mvl, v.Off(n4*q+1-1, n4+1-1), ldv, &epsln, &sfmin, &tol, func() *int { y := 1; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

			Zgsvj1(jobv, &n2, &n2, &n4, a, lda, cwork, sva, &mvl, v, ldv, &epsln, &sfmin, &tol, func() *int { y := 1; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

			Zgsvj0(jobv, toPtr(n2+n4), &n4, a.Off(0, n2+1-1), lda, cwork.Off(n2+1-1), sva.Off(n2+1-1), &mvl, v.Off(n2*q+1-1, n2+1-1), ldv, &epsln, &sfmin, &tol, func() *int { y := 1; return &y }(), cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
		}

	}

	//     .. Row-cyclic pivot strategy with de Rijk's pivoting ..
	for i = 1; i <= nsweep; i++ {
		//     .. go go go ...
		mxaapq = zero
		mxsinj = zero
		iswrot = 0

		notrot = 0
		pskipped = 0

		//     Each sweep is unrolled using KBL-by-KBL tiles over the pivot pairs
		//     1 <= p < q <= N. This is the first step toward a blocked implementation
		//     of the rotations. New implementation, based on block transformations,
		//     is under development.
		for ibr = 1; ibr <= nbl; ibr++ {

			igl = (ibr-1)*kbl + 1

			for ir1 = 0; ir1 <= minint(lkahead, nbl-ibr); ir1++ {

				igl = igl + ir1*kbl

				for p = igl; p <= minint(igl+kbl-1, (*n)-1); p++ {
					//     .. de Rijk's pivoting
					q = goblas.Idamax((*n)-p+1, sva.Off(p-1), 1) + p - 1
					if p != q {
						goblas.Zswap(*m, a.CVector(0, p-1), 1, a.CVector(0, q-1), 1)
						if rsvec {
							goblas.Zswap(mvl, v.CVector(0, p-1), 1, v.CVector(0, q-1), 1)
						}
						temp1 = sva.Get(p - 1)
						sva.Set(p-1, sva.Get(q-1))
						sva.Set(q-1, temp1)
						aapq = cwork.Get(p - 1)
						cwork.Set(p-1, cwork.Get(q-1))
						cwork.Set(q-1, aapq)
					}

					if ir1 == 0 {
						//        Column norms are periodically updated by explicit
						//        norm computation.
						//[!]     Caveat:
						//        Unfortunately, some BLAS implementations compute DZNRM2(M,A(1,p),1)
						//        as SQRT(S=CDOTC(M,A(1,p),1,A(1,p),1)), which may cause the result to
						//        overflow for ||A(:,p)||_2 > SQRT(overflow_threshold), and to
						//        underflow for ||A(:,p)||_2 < SQRT(underflow_threshold).
						//        Hence, DZNRM2 cannot be trusted, not even in the case when
						//        the true norm is far from the under(over)flow boundaries.
						//        If properly implemented SCNRM2 is available, the IF-THEN-ELSE-END IF
						//        below should be replaced with "AAPP = DZNRM2( M, A(1,p), 1 )".
						if (sva.Get(p-1) < rootbig) && (sva.Get(p-1) > rootsfmin) {
							sva.Set(p-1, goblas.Dznrm2(*m, a.CVector(0, p-1), 1))
						} else {
							temp1 = zero
							aapp = one
							Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &temp1, &aapp)
							sva.Set(p-1, temp1*math.Sqrt(aapp))
						}
						aapp = sva.Get(p - 1)
					} else {
						aapp = sva.Get(p - 1)
					}

					if aapp > zero {

						pskipped = 0

						for q = p + 1; q <= minint(igl+kbl-1, *n); q++ {

							aaqq = sva.Get(q - 1)

							if aaqq > zero {

								aapp0 = aapp
								if aaqq >= one {
									rotok = (small * aapp) <= aaqq
									if aapp < (big / aaqq) {
										aapq = (goblas.Zdotc(*m, a.CVector(0, p-1), 1, a.CVector(0, q-1), 1) / complex(aaqq, 0)) / complex(aapp, 0)
									} else {
										goblas.Zcopy(*m, a.CVector(0, p-1), 1, cwork.Off((*n)+1-1), 1)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), cwork.CMatrixOff((*n)+1-1, *lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(*m, cwork.Off((*n)+1-1), 1, a.CVector(0, q-1), 1) / complex(aaqq, 0)
									}
								} else {
									rotok = aapp <= (aaqq / small)
									if aapp > (small / aaqq) {
										aapq = (goblas.Zdotc(*m, a.CVector(0, p-1), 1, a.CVector(0, q-1), 1) / complex(aapp, 0)) / complex(aaqq, 0)
									} else {
										goblas.Zcopy(*m, a.CVector(0, q-1), 1, cwork.Off((*n)+1-1), 1)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), cwork.CMatrixOff((*n)+1-1, *lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(*m, a.CVector(0, p-1), 1, cwork.Off((*n)+1-1), 1) / complex(aapp, 0)
									}
								}

								//                           AAPQ = AAPQ * CONJG( CWORK(p) ) * CWORK(q)
								aapq1 = -cmplx.Abs(aapq)
								mxaapq = maxf64(mxaapq, -aapq1)

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq1) > tol {
									ompq = aapq / complex(cmplx.Abs(aapq), 0)

									//           .. rotate
									//[RTD]      ROTATED = ROTATED + ONE
									if ir1 == 0 {
										notrot = 0
										pskipped = 0
										iswrot = iswrot + 1
									}

									if rotok {

										aqoap = aaqq / aapp
										apoaq = aapp / aaqq
										theta = -half * math.Abs(aqoap-apoaq) / aapq1

										if math.Abs(theta) > bigtheta {

											t = half / theta
											cs = one
											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											}
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))
											mxsinj = maxf64(mxsinj, math.Abs(t))

										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq1)
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs

											mxsinj = maxf64(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))

											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											}
										}
										cwork.Set(p-1, -cwork.Get(q-1)*ompq)

									} else {
										//              .. have to use modified Gram-Schmidt like transformation
										goblas.Zcopy(*m, a.CVector(0, p-1), 1, cwork.Off((*n)+1-1), 1)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), cwork.CMatrixOff((*n)+1-1, *lda, opts), lda, &ierr)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
										goblas.Zaxpy(*m, -aapq, cwork.Off((*n)+1-1), 1, a.CVector(0, q-1), 1)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aaqq, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
										sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one-aapq1*aapq1)))
										mxsinj = maxf64(mxsinj, sfmin)
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q), SVA(p)
									//           recompute SVA(q), SVA(p).
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dznrm2(*m, a.CVector(0, q-1), 1))
										} else {
											t = zero
											aaqq = one
											Zlassq(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq))
										}
									}
									if (aapp / aapp0) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dznrm2(*m, a.CVector(0, p-1), 1)
										} else {
											t = zero
											aapp = one
											Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &t, &aapp)
											aapp = t * math.Sqrt(aapp)
										}
										sva.Set(p-1, aapp)
									}

								} else {
									//                             A(:,p) and A(:,q) already numerically orthogonal
									if ir1 == 0 {
										notrot = notrot + 1
									}
									//[RTD]      SKIPPED  = SKIPPED + 1
									pskipped = pskipped + 1
								}
							} else {
								//                          A(:,q) is zero column
								if ir1 == 0 {
									notrot = notrot + 1
								}
								pskipped = pskipped + 1
							}

							if (i <= swband) && (pskipped > rowskip) {
								if ir1 == 0 {
									aapp = -aapp
								}
								notrot = 0
								goto label2103
							}

						}
						//     END q-LOOP

					label2103:
						;
						//     bailed out of q-loop
						sva.Set(p-1, aapp)

					} else {
						sva.Set(p-1, aapp)
						if (ir1 == 0) && (aapp == zero) {
							notrot = notrot + minint(igl+kbl-1, *n) - p
						}
					}

				}
				//     end of the p-loop
				//     end of doing the block ( ibr, ibr )
			}
			//     end of ir1-loop
			//
			// ... go to the off diagonal blocks
			igl = (ibr-1)*kbl + 1

			for jbc = ibr + 1; jbc <= nbl; jbc++ {

				jgl = (jbc-1)*kbl + 1

				//        doing the block at ( ibr, jbc )
				ijblsk = 0
				for p = igl; p <= minint(igl+kbl-1, *n); p++ {

					aapp = sva.Get(p - 1)
					if aapp > zero {

						pskipped = 0

						for q = jgl; q <= minint(jgl+kbl-1, *n); q++ {

							aaqq = sva.Get(q - 1)
							if aaqq > zero {
								aapp0 = aapp

								//     .. M x 2 Jacobi SVD ..
								//
								//        Safe Gram matrix computation
								if aaqq >= one {
									if aapp >= aaqq {
										rotok = (small * aapp) <= aaqq
									} else {
										rotok = (small * aaqq) <= aapp
									}
									if aapp < (big / aaqq) {
										aapq = (goblas.Zdotc(*m, a.CVector(0, p-1), 1, a.CVector(0, q-1), 1) / complex(aaqq, 0)) / complex(aapp, 0)
									} else {
										goblas.Zcopy(*m, a.CVector(0, p-1), 1, cwork.Off((*n)+1-1), 1)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), cwork.CMatrixOff((*n)+1-1, *lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(*m, cwork.Off((*n)+1-1), 1, a.CVector(0, q-1), 1) / complex(aaqq, 0)
									}
								} else {
									if aapp >= aaqq {
										rotok = aapp <= (aaqq / small)
									} else {
										rotok = aaqq <= (aapp / small)
									}
									if aapp > (small / aaqq) {
										aapq = (goblas.Zdotc(*m, a.CVector(0, p-1), 1, a.CVector(0, q-1), 1) / complex(maxf64(aaqq, aapp), 0)) / complex(minf64(aaqq, aapp), 0)
									} else {
										goblas.Zcopy(*m, a.CVector(0, q-1), 1, cwork.Off((*n)+1-1), 1)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), cwork.CMatrixOff((*n)+1-1, *lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(*m, a.CVector(0, p-1), 1, cwork.Off((*n)+1-1), 1) / complex(aapp, 0)
									}
								}

								//                           AAPQ = AAPQ * CONJG(CWORK(p))*CWORK(q)
								aapq1 = -cmplx.Abs(aapq)
								mxaapq = maxf64(mxaapq, -aapq1)

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq1) > tol {
									ompq = aapq / complex(cmplx.Abs(aapq), 0)
									notrot = 0
									//[RTD]      ROTATED  = ROTATED + 1
									pskipped = 0
									iswrot = iswrot + 1

									if rotok {

										aqoap = aaqq / aapp
										apoaq = aapp / aaqq
										theta = -half * math.Abs(aqoap-apoaq) / aapq1
										if aaqq > aapp0 {
											theta = -theta
										}

										if math.Abs(theta) > bigtheta {
											t = half / theta
											cs = one
											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											}
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))
											mxsinj = maxf64(mxsinj, math.Abs(t))
										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq1)
											if aaqq > aapp0 {
												thsign = -thsign
											}
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs
											mxsinj = maxf64(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))

											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											}
										}
										cwork.Set(p-1, -cwork.Get(q-1)*ompq)

									} else {
										//              .. have to use modified Gram-Schmidt like transformation
										if aapp > aaqq {
											goblas.Zcopy(*m, a.CVector(0, p-1), 1, cwork.Off((*n)+1-1), 1)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), cwork.CMatrixOff((*n)+1-1, *lda, opts), lda, &ierr)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
											goblas.Zaxpy(*m, -aapq, cwork.Off((*n)+1-1), 1, a.CVector(0, q-1), 1)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aaqq, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one-aapq1*aapq1)))
											mxsinj = maxf64(mxsinj, sfmin)
										} else {
											goblas.Zcopy(*m, a.CVector(0, q-1), 1, cwork.Off((*n)+1-1), 1)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), cwork.CMatrixOff((*n)+1-1, *lda, opts), lda, &ierr)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), a.Off(0, p-1), lda, &ierr)
											goblas.Zaxpy(*m, -cmplx.Conj(aapq), cwork.Off((*n)+1-1), 1, a.CVector(0, p-1), 1)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aapp, m, func() *int { y := 1; return &y }(), a.Off(0, p-1), lda, &ierr)
											sva.Set(p-1, aapp*math.Sqrt(maxf64(zero, one-aapq1*aapq1)))
											mxsinj = maxf64(mxsinj, sfmin)
										}
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q), SVA(p)
									//           .. recompute SVA(q), SVA(p)
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dznrm2(*m, a.CVector(0, q-1), 1))
										} else {
											t = zero
											aaqq = one
											Zlassq(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq))
										}
									}
									if math.Pow(aapp/aapp0, 2) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dznrm2(*m, a.CVector(0, p-1), 1)
										} else {
											t = zero
											aapp = one
											Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &t, &aapp)
											aapp = t * math.Sqrt(aapp)
										}
										sva.Set(p-1, aapp)
									}
									//              end of OK rotation
								} else {
									notrot = notrot + 1
									//[RTD]      SKIPPED  = SKIPPED  + 1
									pskipped = pskipped + 1
									ijblsk = ijblsk + 1
								}
							} else {
								notrot = notrot + 1
								pskipped = pskipped + 1
								ijblsk = ijblsk + 1
							}

							if (i <= swband) && (ijblsk >= blskip) {
								sva.Set(p-1, aapp)
								notrot = 0
								goto label2011
							}
							if (i <= swband) && (pskipped > rowskip) {
								aapp = -aapp
								notrot = 0
								goto label2203
							}

						}
						//        end of the q-loop
					label2203:
						;

						sva.Set(p-1, aapp)

					} else {

						if aapp == zero {
							notrot = notrot + minint(jgl+kbl-1, *n) - jgl + 1
						}
						if aapp < zero {
							notrot = 0
						}

					}

				}
				//     end of the p-loop
			}
			//     end of the jbc-loop
		label2011:
			;
			//2011 bailed out of the jbc-loop
			for p = igl; p <= minint(igl+kbl-1, *n); p++ {
				sva.Set(p-1, sva.GetMag(p-1))
			}
			//**
		}
		//2000 :: end of the ibr-loop
		//
		//     .. update SVA(N)
		if (sva.Get((*n)-1) < rootbig) && (sva.Get((*n)-1) > rootsfmin) {
			sva.Set((*n)-1, goblas.Dznrm2(*m, a.CVector(0, (*n)-1), 1))
		} else {
			t = zero
			aapp = one
			Zlassq(m, a.CVector(0, (*n)-1), func() *int { y := 1; return &y }(), &t, &aapp)
			sva.Set((*n)-1, t*math.Sqrt(aapp))
		}

		//     Additional steering devices
		if (i < swband) && ((mxaapq <= roottol) || (iswrot <= (*n))) {
			swband = i
		}

		if (i > swband+1) && (mxaapq < math.Sqrt(float64(*n))*tol) && (float64(*n)*mxaapq*mxsinj < tol) {
			goto label1994
		}

		if notrot >= emptsw {
			goto label1994
		}

	}
	//     end i=1:NSWEEP loop
	//
	// #:( Reaching this point means that the procedure has not converged.
	(*info) = nsweep - 1
	goto label1995

label1994:
	;
	// #:) Reaching this point means numerical convergence after the i-th
	//     sweep.
	//
	(*info) = 0
	// #:) INFO = 0 confirms successful iterations.
label1995:
	;

	//     Sort the singular values and find how many are above
	//     the underflow threshold.
	n2 = 0
	n4 = 0
	for p = 1; p <= (*n)-1; p++ {
		q = goblas.Idamax((*n)-p+1, sva.Off(p-1), 1) + p - 1
		if p != q {
			temp1 = sva.Get(p - 1)
			sva.Set(p-1, sva.Get(q-1))
			sva.Set(q-1, temp1)
			goblas.Zswap(*m, a.CVector(0, p-1), 1, a.CVector(0, q-1), 1)
			if rsvec {
				goblas.Zswap(mvl, v.CVector(0, p-1), 1, v.CVector(0, q-1), 1)
			}
		}
		if sva.Get(p-1) != zero {
			n4 = n4 + 1
			if sva.Get(p-1)*skl > sfmin {
				n2 = n2 + 1
			}
		}
	}
	if sva.Get((*n)-1) != zero {
		n4 = n4 + 1
		if sva.Get((*n)-1)*skl > sfmin {
			n2 = n2 + 1
		}
	}

	//     Normalize the left singular vectors.
	if lsvec || uctol {
		for p = 1; p <= n4; p++ {
			//            CALL ZDSCAL( M, ONE / SVA( p ), A( 1, p ), 1 )
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), sva.GetPtr(p-1), &one, m, func() *int { y := 1; return &y }(), a.Off(0, p-1), m, &ierr)
		}
	}

	//     Scale the product of Jacobi rotations.
	if rsvec {
		for p = 1; p <= (*n); p++ {
			temp1 = one / goblas.Dznrm2(mvl, v.CVector(0, p-1), 1)
			goblas.Zdscal(mvl, temp1, v.CVector(0, p-1), 1)
		}
	}

	//     Undo scaling, if necessary (and possible).
	if ((skl > one) && (sva.Get(0) < (big / skl))) || ((skl < one) && (sva.Get(maxint(n2, 1)-1) > (sfmin / skl))) {
		for p = 1; p <= (*n); p++ {
			sva.Set(p-1, skl*sva.Get(p-1))
		}
		skl = one
	}

	rwork.Set(0, skl)
	//     The singular values of A are SKL*SVA(1:N). If SKL.NE.ONE
	//     then some of the singular values may overflow or underflow and
	//     the spectrum is given in this factored representation.
	rwork.Set(1, float64(n4))
	//     N4 is the number of computed nonzero singular values of A.

	rwork.Set(2, float64(n2))
	//     N2 is the number of singular values of A greater than SFMIN.
	//     If N2<N, SVA(N2:N) contains ZEROS and/or denormalized numbers
	//     that may carry some information.

	rwork.Set(3, float64(i))
	//     i is the index of the last sweep before declaring convergence.

	rwork.Set(4, mxaapq)
	//     MXAAPQ is the largest absolute value of scaled pivots in the
	//     last sweep

	rwork.Set(5, mxsinj)
	//     MXSINJ is the largest absolute value of the sines of Jacobi angles
	//     in the last sweep
}
