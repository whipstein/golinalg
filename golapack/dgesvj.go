package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgesvj computes the singular value decomposition (SVD) of a real
// M-by-N matrix A, where M >= N. The SVD of A is written as
//                                    [++]   [xx]   [x0]   [xx]
//              A = U * SIGMA * V^t,  [++] = [xx] * [ox] * [xx]
//                                    [++]   [xx]
// where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
// matrix, and V is an N-by-N orthogonal matrix. The diagonal elements
// of SIGMA are the singular values of A. The columns of U and V are the
// left and the right singular vectors of A, respectively.
// Dgesvj can sometimes compute tiny singular values and their singular vectors much
// more accurately than other SVD routines, see below under Further Details.
func Dgesvj(joba, jobu, jobv byte, m, n int, a *mat.Matrix, sva *mat.Vector, mv int, v *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var applv, goscale, lower, lsvec, noscale, rotok, rsvec, uctol, upper bool
	var aapp, aapp0, aapq, aaqq, apoaq, aqoap, big, bigtheta, cs, ctol, epsln, half, mxaapq, mxsinj, one, rootbig, rooteps, rootsfmin, roottol, sfmin, skl, small, sn, t, temp1, theta, thsign, tol, zero float64
	var blskip, emptsw, i, ibr, igl, ijblsk, ir1, iswrot, jbc, jgl, kbl, lkahead, mvl, n2, n34, n4, nbl, notrot, nsweep, p, pskipped, q, rowskip, swband int

	fastr := mat.NewDrotMatrix()

	zero = 0.0
	half = 0.5
	one = 1.0
	nsweep = 30

	//     Test the input arguments
	lsvec = jobu == 'U'
	uctol = jobu == 'C'
	rsvec = jobv == 'V'
	applv = jobv == 'A'
	upper = joba == 'U'
	lower = joba == 'L'

	if !(upper || lower || joba == 'G') {
		err = fmt.Errorf("!(upper || lower || joba == 'G'): joba='%c'", joba)
	} else if !(lsvec || uctol || jobu == 'N') {
		err = fmt.Errorf("!(lsvec || uctol || jobu == 'N'): jobu='%c'", jobu)
	} else if !(rsvec || applv || jobv == 'N') {
		err = fmt.Errorf("!(rsvec || applv || jobv == 'N'): jobv='%c'", jobv)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if (n < 0) || (n > m) {
		err = fmt.Errorf("(n < 0) || (n > m): m=%v, n=%v", m, n)
	} else if a.Rows < m {
		err = fmt.Errorf("a.Rows < m: a.Rows=%v, m=%v", a.Rows, m)
	} else if mv < 0 {
		err = fmt.Errorf("mv < 0: mv=%v", mv)
	} else if (rsvec && (v.Rows < n)) || (applv && (v.Rows < mv)) {
		err = fmt.Errorf("(rsvec && (v.Rows < n)) || (applv && (v.Rows < mv)): jobv='%c', v.Rows=%v, n=%v, mv=%v", jobv, v.Rows, n, mv)
	} else if uctol && (work.Get(0) <= one) {
		err = fmt.Errorf("uctol && (work.Get(0) <= one): jobu='%c', work(0)=%v", jobu, work.Get(0))
	} else if lwork < max(m+n, 6) {
		err = fmt.Errorf("lwork < max(m+n, 6): lwork=%v, m=%v, n=%v", lwork, m, n)
	}

	//     #:(
	if err != nil {
		gltest.Xerbla2("Dgesvj", err)
		return
	}

	// #:) Quick return for void matrix
	if (m == 0) || (n == 0) {
		return
	}

	//     Set numerical parameters
	//     The stopping criterion for Jacobi rotations is
	//
	//     max_{i<>j}|A(:,i)^T * A(:,j)|/(||A(:,i)||*||A(:,j)||) < CTOL*EPS
	//
	//     where EPS is the round-off and CTOL is defined as follows:
	if uctol {
		//        ... user controlled
		ctol = work.Get(0)
	} else {
		//        ... default
		if lsvec || rsvec || applv {
			ctol = math.Sqrt(float64(m))
		} else {
			ctol = float64(m)
		}
	}
	//     ... and the machine dependent parameters are
	//[!]  (Make sure that DLAMCH() works properly on the target machine.)
	epsln = Dlamch(Epsilon)
	rooteps = math.Sqrt(epsln)
	sfmin = Dlamch(SafeMinimum)
	rootsfmin = math.Sqrt(sfmin)
	small = sfmin / epsln
	big = Dlamch(Overflow)
	//     BIG         = ONE    / SFMIN
	rootbig = one / rootsfmin
	// large = big / math.Sqrt(float64(m*n))
	bigtheta = one / rooteps

	tol = ctol * epsln
	roottol = math.Sqrt(tol)

	if float64(m)*epsln >= one {
		err = fmt.Errorf("float64(m)*epsln >= one: m=%v, epsln=%v", m, epsln)
		gltest.Xerbla2("Dgesvj", err)
		return
	}

	//     Initialize the right singular vector matrix.
	if rsvec {
		mvl = n
		Dlaset(Full, mvl, n, zero, one, v)
	} else if applv {
		mvl = mv
	}
	rsvec = rsvec || applv

	//     Initialize SVA( 1:N ) = ( ||A e_i||_2, i = 1:N )
	//(!)  If necessary, scale A to protect the largest singular value
	//     from overflow. It is possible that saving the largest singular
	//     value destroys the information about the small ones.
	//     This initial scaling is almost minimal in the sense that the
	//     goal is to make sure that no column norm overflows, and that
	//     DSQRT(N)*max_i SVA(i) does not overflow. If INFinite entries
	//     in A are detected, the procedure returns with INFO=-6.
	skl = one / math.Sqrt(float64(m)*float64(n))
	noscale = true
	goscale = true

	if lower {
		//        the input matrix is M-by-N lower triangular (trapezoidal)
		for p = 1; p <= n; p++ {
			aapp = zero
			aaqq = one
			aapp, aaqq = Dlassq(m-p+1, a.Off(p-1, p-1).Vector(), 1, aapp, aaqq)
			if aapp > big {
				err = fmt.Errorf("aapp > big: aapp=%v", aapp)
				gltest.Xerbla2("Dgesvj", err)
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
		for p = 1; p <= n; p++ {
			aapp = zero
			aaqq = one
			aapp, aaqq = Dlassq(p, a.Off(0, p-1).Vector(), 1, aapp, aaqq)
			if aapp > big {
				err = fmt.Errorf("aapp > big: aapp=%v", aapp)
				gltest.Xerbla2("Dgesvj", err)
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
		for p = 1; p <= n; p++ {
			aapp = zero
			aaqq = one
			aapp, aaqq = Dlassq(m, a.Off(0, p-1).Vector(), 1, aapp, aaqq)
			if aapp > big {
				err = fmt.Errorf("aapp > big: aapp=%v", aapp)
				gltest.Xerbla2("Dgesvj", err)
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
	for p = 1; p <= n; p++ {
		if sva.Get(p-1) != zero {
			aaqq = math.Min(aaqq, sva.Get(p-1))
		}
		aapp = math.Max(aapp, sva.Get(p-1))
	}

	// #:) Quick return for zero matrix
	if aapp == zero {
		if lsvec {
			Dlaset(Full, m, n, zero, one, a)
		}
		work.Set(0, one)
		work.Set(1, zero)
		work.Set(2, zero)
		work.Set(3, zero)
		work.Set(4, zero)
		work.Set(5, zero)
		return
	}

	// #:) Quick return for one-column matrix
	if n == 1 {
		if lsvec {
			if err = Dlascl('G', 0, 0, sva.Get(0), skl, m, 1, a); err != nil {
				panic(err)
			}
		}
		work.Set(0, one/skl)
		if sva.Get(0) >= sfmin {
			work.Set(1, one)
		} else {
			work.Set(1, zero)
		}
		work.Set(2, zero)
		work.Set(3, zero)
		work.Set(4, zero)
		work.Set(5, zero)
		return
	}

	//     Protect small singular values from underflow, and try to
	//     avoid underflows/overflows in computing Jacobi rotations.
	sn = math.Sqrt(sfmin / epsln)
	temp1 = math.Sqrt(big / float64(n))
	if (aapp <= sn) || (aaqq >= temp1) || ((sn <= aaqq) && (aapp <= temp1)) {
		temp1 = math.Min(big, temp1/aapp)
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else if (aaqq <= sn) && (aapp <= temp1) {
		temp1 = math.Min(sn/aaqq, big/(aapp*math.Sqrt(float64(n))))
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else if (aaqq >= sn) && (aapp >= temp1) {
		temp1 = math.Max(sn/aaqq, temp1/aapp)
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else if (aaqq <= sn) && (aapp >= temp1) {
		temp1 = math.Min(sn/aaqq, big/(math.Sqrt(float64(n))*aapp))
		//         AAQQ  = AAQQ*TEMP1
		//         AAPP  = AAPP*TEMP1
	} else {
		temp1 = one
	}

	//     Scale, if necessary
	if temp1 != one {
		if err = Dlascl('G', 0, 0, one, temp1, n, 1, sva.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}
	skl = temp1 * skl
	if skl != one {
		if err = Dlascl(joba, 0, 0, one, skl, m, n, a); err != nil {
			panic(err)
		}
		skl = one / skl
	}

	//     Row-cyclic Jacobi SVD algorithm with column pivoting
	emptsw = (n * (n - 1)) / 2
	notrot = 0
	fastr.Flag = int(zero)

	//     A is represented in factored form A = A * diag(WORK), where diag(WORK)
	//     is initialized to identity. WORK is updated during fast scaled
	//     rotations.
	for q = 1; q <= n; q++ {
		work.Set(q-1, one)
	}

	swband = 3
	//[TP] SWBAND is a tuning parameter [TP]. It is meaningful and effective
	//     if Dgesvj is used as a computational routine in the preconditioned
	//     Jacobi SVD algorithm Dgesvj. For sweeps i=1:SWBAND the procedure
	//     works on pivots inside a band-like region around the diagonal.
	//     The boundaries are determined dynamically, based on the number of
	//     pivots above a threshold.

	kbl = min(int(8), n)
	//[TP] KBL is a tuning parameter that defines the tile size in the
	//     tiling of the p-q loops of pivot pairs. In general, an optimal
	//     value of KBL depends on the matrix dimensions and on the
	//     parameters of the computer's memory.

	nbl = n / kbl
	if (nbl * kbl) != n {
		nbl = nbl + 1
	}

	blskip = int(math.Pow(float64(kbl), 2))
	//[TP] BLKSKIP is a tuning parameter that depends on SWBAND and KBL.

	rowskip = min(int(5), kbl)
	//[TP] ROWSKIP is a tuning parameter.

	lkahead = 1
	//[TP] LKAHEAD is a tuning parameter.
	//
	//     Quasi block transformations, using the lower (upper) triangular
	//     structure of the input matrix. The quasi-block-cycling usually
	//     invokes cubic convergence. Big part of this cycle is done inside
	//     canonical subspaces of dimensions less than M.

	if (lower || upper) && (n > max(64, 4*kbl)) {
		//[TP] The number of partition levels and the actual partition are
		//     tuning parameters.
		n4 = n / 4
		n2 = n / 2
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
			if _, err = Dgsvj0(jobv, m-n34, n-n34, a.Off(n34, n34), work.Off(n34), sva.Off(n34), mvl, v.Off(n34*q, n34), epsln, sfmin, tol, 2, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj0(jobv, m-n2, n34-n2, a.Off(n2, n2), work.Off(n2), sva.Off(n2), mvl, v.Off(n2*q, n2), epsln, sfmin, tol, 2, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj1(jobv, m-n2, n-n2, n4, a.Off(n2, n2), work.Off(n2), sva.Off(n2), mvl, v.Off(n2*q, n2), epsln, sfmin, tol, 1, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj0(jobv, m-n4, n2-n4, a.Off(n4, n4), work.Off(n4), sva.Off(n4), mvl, v.Off(n4*q, n4), epsln, sfmin, tol, 1, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj0(jobv, m, n4, a, work, sva, mvl, v, epsln, sfmin, tol, 1, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj1(jobv, m, n2, n4, a, work, sva, mvl, v, epsln, sfmin, tol, 1, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

		} else if upper {
			if _, err = Dgsvj0(jobv, n4, n4, a, work, sva, mvl, v, epsln, sfmin, tol, 2, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj0(jobv, n2, n4, a.Off(0, n4), work.Off(n4), sva.Off(n4), mvl, v.Off(n4*q, n4), epsln, sfmin, tol, 1, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj1(jobv, n2, n2, n4, a, work, sva, mvl, v, epsln, sfmin, tol, 1, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

			if _, err = Dgsvj0(jobv, n2+n4, n4, a.Off(0, n2), work.Off(n2), sva.Off(n2), mvl, v.Off(n2*q, n2), epsln, sfmin, tol, 1, work.Off(n), lwork-n); err != nil {
				panic(err)
			}
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

			for ir1 = 0; ir1 <= min(lkahead, nbl-ibr); ir1++ {

				igl = igl + ir1*kbl

				for p = igl; p <= min(igl+kbl-1, n-1); p++ {
					//     .. de Rijk's pivoting
					q = sva.Off(p-1).Iamax(n-p+1, 1) + p - 1
					if p != q {
						a.Off(0, q-1).Vector().Swap(m, a.Off(0, p-1).Vector(), 1, 1)
						if rsvec {
							v.Off(0, q-1).Vector().Swap(mvl, v.Off(0, p-1).Vector(), 1, 1)
						}
						temp1 = sva.Get(p - 1)
						sva.Set(p-1, sva.Get(q-1))
						sva.Set(q-1, temp1)
						temp1 = work.Get(p - 1)
						work.Set(p-1, work.Get(q-1))
						work.Set(q-1, temp1)
					}

					if ir1 == 0 {
						//        Column norms are periodically updated by explicit
						//        norm computation.
						//        Caveat:
						//        Unfortunately, some BLAS implementations compute DNRM2(M,A(1,p),1)
						//        as DSQRT(DDOT(M,A(1,p),1,A(1,p),1)), which may cause the result to
						//        overflow for ||A(:,p)||_2 > DSQRT(overflow_threshold), and to
						//        underflow for ||A(:,p)||_2 < DSQRT(underflow_threshold).
						//        Hence, DNRM2 cannot be trusted, not even in the case when
						//        the true norm is far from the under(over)flow boundaries.
						//        If properly implemented DNRM2 is available, the IF-THEN-ELSE
						//        below should read "AAPP = DNRM2( M, A(1,p), 1 ) * WORK(p)".
						if (sva.Get(p-1) < rootbig) && (sva.Get(p-1) > rootsfmin) {
							sva.Set(p-1, a.Off(0, p-1).Vector().Nrm2(m, 1)*work.Get(p-1))
						} else {
							temp1 = zero
							aapp = one
							temp1, aapp = Dlassq(m, a.Off(0, p-1).Vector(), 1, temp1, aapp)
							sva.Set(p-1, temp1*math.Sqrt(aapp)*work.Get(p-1))
						}
						aapp = sva.Get(p - 1)
					} else {
						aapp = sva.Get(p - 1)
					}

					if aapp > zero {

						pskipped = 0

						for q = p + 1; q <= min(igl+kbl-1, n); q++ {

							aaqq = sva.Get(q - 1)

							if aaqq > zero {

								aapp0 = aapp
								if aaqq >= one {
									rotok = (small * aapp) <= aaqq
									if aapp < (big / aaqq) {
										aapq = (a.Off(0, q-1).Vector().Dot(m, a.Off(0, p-1).Vector(), 1, 1) * work.Get(p-1) * work.Get(q-1) / aaqq) / aapp
									} else {
										work.Off(n).Copy(m, a.Off(0, p-1).Vector(), 1, 1)
										if err = Dlascl('G', 0, 0, aapp, work.Get(p-1), m, 1, work.Off(n).Matrix(a.Rows, opts)); err != nil {
											panic(err)
										}
										aapq = a.Off(0, q-1).Vector().Dot(m, work.Off(n), 1, 1) * work.Get(q-1) / aaqq
									}
								} else {
									rotok = aapp <= (aaqq / small)
									if aapp > (small / aaqq) {
										aapq = (a.Off(0, q-1).Vector().Dot(m, a.Off(0, p-1).Vector(), 1, 1) * work.Get(p-1) * work.Get(q-1) / aaqq) / aapp
									} else {
										work.Off(n).Copy(m, a.Off(0, q-1).Vector(), 1, 1)
										if err = Dlascl('G', 0, 0, aaqq, work.Get(q-1), m, 1, work.Off(n).Matrix(a.Rows, opts)); err != nil {
											panic(err)
										}
										aapq = a.Off(0, p-1).Vector().Dot(m, work.Off(n), 1, 1) * work.Get(p-1) / aapp
									}
								}

								mxaapq = math.Max(mxaapq, math.Abs(aapq))

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq) > tol {
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
										theta = -half * math.Abs(aqoap-apoaq) / aapq

										if math.Abs(theta) > bigtheta {

											t = half / theta
											fastr.H21 = t * work.Get(p-1) / work.Get(q-1)
											fastr.H12 = -t * work.Get(q-1) / work.Get(p-1)
											a.Off(0, q-1).Vector().Rotm(m, a.Off(0, p-1).Vector(), 1, 1, fastr)
											if rsvec {
												v.Off(0, q-1).Vector().Rotm(mvl, v.Off(0, p-1).Vector(), 1, 1, fastr)
											}
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq))
											mxsinj = math.Max(mxsinj, math.Abs(t))

										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq)
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs

											mxsinj = math.Max(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq))

											apoaq = work.Get(p-1) / work.Get(q-1)
											aqoap = work.Get(q-1) / work.Get(p-1)
											if work.Get(p-1) >= one {
												if work.Get(q-1) >= one {
													fastr.H21 = t * apoaq
													fastr.H12 = -t * aqoap
													work.Set(p-1, work.Get(p-1)*cs)
													work.Set(q-1, work.Get(q-1)*cs)
													a.Off(0, q-1).Vector().Rotm(m, a.Off(0, p-1).Vector(), 1, 1, fastr)
													if rsvec {
														v.Off(0, q-1).Vector().Rotm(mvl, v.Off(0, p-1).Vector(), 1, 1, fastr)
													}
												} else {
													a.Off(0, p-1).Vector().Axpy(m, -t*aqoap, a.Off(0, q-1).Vector(), 1, 1)
													a.Off(0, q-1).Vector().Axpy(m, cs*sn*apoaq, a.Off(0, p-1).Vector(), 1, 1)
													work.Set(p-1, work.Get(p-1)*cs)
													work.Set(q-1, work.Get(q-1)/cs)
													if rsvec {
														v.Off(0, p-1).Vector().Axpy(mvl, -t*aqoap, v.Off(0, q-1).Vector(), 1, 1)
														v.Off(0, q-1).Vector().Axpy(mvl, cs*sn*apoaq, v.Off(0, p-1).Vector(), 1, 1)
													}
												}
											} else {
												if work.Get(q-1) >= one {
													a.Off(0, q-1).Vector().Axpy(m, t*apoaq, a.Off(0, p-1).Vector(), 1, 1)
													a.Off(0, p-1).Vector().Axpy(m, -cs*sn*aqoap, a.Off(0, q-1).Vector(), 1, 1)
													work.Set(p-1, work.Get(p-1)/cs)
													work.Set(q-1, work.Get(q-1)*cs)
													if rsvec {
														v.Off(0, q-1).Vector().Axpy(mvl, t*apoaq, v.Off(0, p-1).Vector(), 1, 1)
														v.Off(0, p-1).Vector().Axpy(mvl, -cs*sn*aqoap, v.Off(0, q-1).Vector(), 1, 1)
													}
												} else {
													if work.Get(p-1) >= work.Get(q-1) {
														a.Off(0, p-1).Vector().Axpy(m, -t*aqoap, a.Off(0, q-1).Vector(), 1, 1)
														a.Off(0, q-1).Vector().Axpy(m, cs*sn*apoaq, a.Off(0, p-1).Vector(), 1, 1)
														work.Set(p-1, work.Get(p-1)*cs)
														work.Set(q-1, work.Get(q-1)/cs)
														if rsvec {
															v.Off(0, p-1).Vector().Axpy(mvl, -t*aqoap, v.Off(0, q-1).Vector(), 1, 1)
															v.Off(0, q-1).Vector().Axpy(mvl, cs*sn*apoaq, v.Off(0, p-1).Vector(), 1, 1)
														}
													} else {
														a.Off(0, q-1).Vector().Axpy(m, t*apoaq, a.Off(0, p-1).Vector(), 1, 1)
														a.Off(0, p-1).Vector().Axpy(m, -cs*sn*aqoap, a.Off(0, q-1).Vector(), 1, 1)
														work.Set(p-1, work.Get(p-1)/cs)
														work.Set(q-1, work.Get(q-1)*cs)
														if rsvec {
															v.Off(0, q-1).Vector().Axpy(mvl, t*apoaq, v.Off(0, p-1).Vector(), 1, 1)
															v.Off(0, p-1).Vector().Axpy(mvl, -cs*sn*aqoap, v.Off(0, q-1).Vector(), 1, 1)
														}
													}
												}
											}
										}

									} else {
										//              .. have to use modified Gram-Schmidt like transformation
										work.Off(n).Copy(m, a.Off(0, p-1).Vector(), 1, 1)
										if err = Dlascl('G', 0, 0, aapp, one, m, 1, work.Off(n).Matrix(a.Rows, opts)); err != nil {
											panic(err)
										}
										if err = Dlascl('G', 0, 0, aaqq, one, m, 1, a.Off(0, q-1)); err != nil {
											panic(err)
										}
										temp1 = -aapq * work.Get(p-1) / work.Get(q-1)
										a.Off(0, q-1).Vector().Axpy(m, temp1, work.Off(n), 1, 1)
										if err = Dlascl('G', 0, 0, one, aaqq, m, 1, a.Off(0, q-1)); err != nil {
											panic(err)
										}
										sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one-aapq*aapq)))
										mxsinj = math.Max(mxsinj, sfmin)
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q), SVA(p)
									//           recompute SVA(q), SVA(p).

									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, a.Off(0, q-1).Vector().Nrm2(m, 1)*work.Get(q-1))
										} else {
											t = zero
											aaqq = one
											t, aaqq = Dlassq(m, a.Off(0, q-1).Vector(), 1, t, aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq)*work.Get(q-1))
										}
									}
									if (aapp / aapp0) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = a.Off(0, p-1).Vector().Nrm2(m, 1) * work.Get(p-1)
										} else {
											t = zero
											aapp = one
											t, aapp = Dlassq(m, a.Off(0, p-1).Vector(), 1, t, aapp)
											aapp = t * math.Sqrt(aapp) * work.Get(p-1)
										}
										sva.Set(p-1, aapp)
									}

								} else {
									//        A(:,p) and A(:,q) already numerically orthogonal
									if ir1 == 0 {
										notrot = notrot + 1
									}
									//[RTD]      SKIPPED  = SKIPPED  + 1
									pskipped = pskipped + 1
								}
							} else {
								//        A(:,q) is zero column
								if ir1 == 0 {
									notrot = notrot + 1
								}
								pskipped = pskipped + 1
							}
							//
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
							notrot = notrot + min(igl+kbl-1, n) - p
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
				for p = igl; p <= min(igl+kbl-1, n); p++ {

					aapp = sva.Get(p - 1)
					if aapp > zero {

						pskipped = 0

						for q = jgl; q <= min(jgl+kbl-1, n); q++ {

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
										aapq = (a.Off(0, q-1).Vector().Dot(m, a.Off(0, p-1).Vector(), 1, 1) * work.Get(p-1) * work.Get(q-1) / aaqq) / aapp
									} else {
										work.Off(n).Copy(m, a.Off(0, p-1).Vector(), 1, 1)
										if err = Dlascl('G', 0, 0, aapp, work.Get(p-1), m, 1, work.Off(n).Matrix(a.Rows, opts)); err != nil {
											panic(err)
										}
										aapq = a.Off(0, q-1).Vector().Dot(m, work.Off(n), 1, 1) * work.Get(q-1) / aaqq
									}
								} else {
									if aapp >= aaqq {
										rotok = aapp <= (aaqq / small)
									} else {
										rotok = aaqq <= (aapp / small)
									}
									if aapp > (small / aaqq) {
										aapq = (a.Off(0, q-1).Vector().Dot(m, a.Off(0, p-1).Vector(), 1, 1) * work.Get(p-1) * work.Get(q-1) / aaqq) / aapp
									} else {
										work.Off(n).Copy(m, a.Off(0, q-1).Vector(), 1, 1)
										if err = Dlascl('G', 0, 0, aaqq, work.Get(q-1), m, 1, work.Off(n).Matrix(a.Rows, opts)); err != nil {
											panic(err)
										}
										aapq = a.Off(0, p-1).Vector().Dot(m, work.Off(n), 1, 1) * work.Get(p-1) / aapp
									}
								}

								mxaapq = math.Max(mxaapq, math.Abs(aapq))

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq) > tol {
									notrot = 0
									//[RTD]      ROTATED  = ROTATED + 1
									pskipped = 0
									iswrot = iswrot + 1

									if rotok {

										aqoap = aaqq / aapp
										apoaq = aapp / aaqq
										theta = -half * math.Abs(aqoap-apoaq) / aapq
										if aaqq > aapp0 {
											theta = -theta
										}

										if math.Abs(theta) > bigtheta {
											t = half / theta
											fastr.H21 = t * work.Get(p-1) / work.Get(q-1)
											fastr.H12 = -t * work.Get(q-1) / work.Get(p-1)
											a.Off(0, q-1).Vector().Rotm(m, a.Off(0, p-1).Vector(), 1, 1, fastr)
											if rsvec {
												v.Off(0, q-1).Vector().Rotm(mvl, v.Off(0, p-1).Vector(), 1, 1, fastr)
											}
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq))
											mxsinj = math.Max(mxsinj, math.Abs(t))
										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq)
											if aaqq > aapp0 {
												thsign = -thsign
											}
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs
											mxsinj = math.Max(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq))

											apoaq = work.Get(p-1) / work.Get(q-1)
											aqoap = work.Get(q-1) / work.Get(p-1)
											if work.Get(p-1) >= one {

												if work.Get(q-1) >= one {
													fastr.H21 = t * apoaq
													fastr.H12 = -t * aqoap
													work.Set(p-1, work.Get(p-1)*cs)
													work.Set(q-1, work.Get(q-1)*cs)
													a.Off(0, q-1).Vector().Rotm(m, a.Off(0, p-1).Vector(), 1, 1, fastr)
													if rsvec {
														v.Off(0, q-1).Vector().Rotm(mvl, v.Off(0, p-1).Vector(), 1, 1, fastr)
													}
												} else {
													a.Off(0, p-1).Vector().Axpy(m, -t*aqoap, a.Off(0, q-1).Vector(), 1, 1)
													a.Off(0, q-1).Vector().Axpy(m, cs*sn*apoaq, a.Off(0, p-1).Vector(), 1, 1)
													if rsvec {
														v.Off(0, p-1).Vector().Axpy(mvl, -t*aqoap, v.Off(0, q-1).Vector(), 1, 1)
														v.Off(0, q-1).Vector().Axpy(mvl, cs*sn*apoaq, v.Off(0, p-1).Vector(), 1, 1)
													}
													work.Set(p-1, work.Get(p-1)*cs)
													work.Set(q-1, work.Get(q-1)/cs)
												}
											} else {
												if work.Get(q-1) >= one {
													a.Off(0, q-1).Vector().Axpy(m, t*apoaq, a.Off(0, p-1).Vector(), 1, 1)
													a.Off(0, p-1).Vector().Axpy(m, -cs*sn*aqoap, a.Off(0, q-1).Vector(), 1, 1)
													if rsvec {
														v.Off(0, q-1).Vector().Axpy(mvl, t*apoaq, v.Off(0, p-1).Vector(), 1, 1)
														v.Off(0, p-1).Vector().Axpy(mvl, -cs*sn*aqoap, v.Off(0, q-1).Vector(), 1, 1)
													}
													work.Set(p-1, work.Get(p-1)/cs)
													work.Set(q-1, work.Get(q-1)*cs)
												} else {
													if work.Get(p-1) >= work.Get(q-1) {
														a.Off(0, p-1).Vector().Axpy(m, -t*aqoap, a.Off(0, q-1).Vector(), 1, 1)
														a.Off(0, q-1).Vector().Axpy(m, cs*sn*apoaq, a.Off(0, p-1).Vector(), 1, 1)
														work.Set(p-1, work.Get(p-1)*cs)
														work.Set(q-1, work.Get(q-1)/cs)
														if rsvec {
															v.Off(0, p-1).Vector().Axpy(mvl, -t*aqoap, v.Off(0, q-1).Vector(), 1, 1)
															v.Off(0, q-1).Vector().Axpy(mvl, cs*sn*apoaq, v.Off(0, p-1).Vector(), 1, 1)
														}
													} else {
														a.Off(0, q-1).Vector().Axpy(m, t*apoaq, a.Off(0, p-1).Vector(), 1, 1)
														a.Off(0, p-1).Vector().Axpy(m, -cs*sn*aqoap, a.Off(0, q-1).Vector(), 1, 1)
														work.Set(p-1, work.Get(p-1)/cs)
														work.Set(q-1, work.Get(q-1)*cs)
														if rsvec {
															v.Off(0, q-1).Vector().Axpy(mvl, t*apoaq, v.Off(0, p-1).Vector(), 1, 1)
															v.Off(0, p-1).Vector().Axpy(mvl, -cs*sn*aqoap, v.Off(0, q-1).Vector(), 1, 1)
														}
													}
												}
											}
										}

									} else {
										if aapp > aaqq {
											work.Off(n).Copy(m, a.Off(0, p-1).Vector(), 1, 1)
											if err = Dlascl('G', 0, 0, aapp, one, m, 1, work.Off(n).Matrix(a.Rows, opts)); err != nil {
												panic(err)
											}
											if err = Dlascl('G', 0, 0, aaqq, one, m, 1, a.Off(0, q-1)); err != nil {
												panic(err)
											}
											temp1 = -aapq * work.Get(p-1) / work.Get(q-1)
											a.Off(0, q-1).Vector().Axpy(m, temp1, work.Off(n), 1, 1)
											if err = Dlascl('G', 0, 0, one, aaqq, m, 1, a.Off(0, q-1)); err != nil {
												panic(err)
											}
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one-aapq*aapq)))
											mxsinj = math.Max(mxsinj, sfmin)
										} else {
											work.Off(n).Copy(m, a.Off(0, q-1).Vector(), 1, 1)
											if err = Dlascl('G', 0, 0, aaqq, one, m, 1, work.Off(n).Matrix(a.Rows, opts)); err != nil {
												panic(err)
											}
											if err = Dlascl('G', 0, 0, aapp, one, m, 1, a.Off(0, p-1)); err != nil {
												panic(err)
											}
											temp1 = -aapq * work.Get(q-1) / work.Get(p-1)
											a.Off(0, p-1).Vector().Axpy(m, temp1, work.Off(n), 1, 1)
											if err = Dlascl('G', 0, 0, one, aapp, m, 1, a.Off(0, p-1)); err != nil {
												panic(err)
											}
											sva.Set(p-1, aapp*math.Sqrt(math.Max(zero, one-aapq*aapq)))
											mxsinj = math.Max(mxsinj, sfmin)
										}
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q)
									//           .. recompute SVA(q)
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, a.Off(0, q-1).Vector().Nrm2(m, 1)*work.Get(q-1))
										} else {
											t = zero
											aaqq = one
											t, aaqq = Dlassq(m, a.Off(0, q-1).Vector(), 1, t, aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq)*work.Get(q-1))
										}
									}
									if math.Pow(aapp/aapp0, 2) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = a.Off(0, p-1).Vector().Nrm2(m, 1) * work.Get(p-1)
										} else {
											t = zero
											aapp = one
											t, aapp = Dlassq(m, a.Off(0, p-1).Vector(), 1, t, aapp)
											aapp = t * math.Sqrt(aapp) * work.Get(p-1)
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
							notrot = notrot + min(jgl+kbl-1, n) - jgl + 1
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
			for p = igl; p <= min(igl+kbl-1, n); p++ {
				sva.Set(p-1, math.Abs(sva.Get(p-1)))
			}
			//**
		}
		//2000 :: end of the ibr-loop
		//
		//     .. update SVA(N)
		if (sva.Get(n-1) < rootbig) && (sva.Get(n-1) > rootsfmin) {
			sva.Set(n-1, a.Off(0, n-1).Vector().Nrm2(m, 1)*work.Get(n-1))
		} else {
			t = zero
			aapp = one
			t, aapp = Dlassq(m, a.Off(0, n-1).Vector(), 1, t, aapp)
			sva.Set(n-1, t*math.Sqrt(aapp)*work.Get(n-1))
		}

		//     Additional steering devices
		if (i < swband) && ((mxaapq <= roottol) || (iswrot <= n)) {
			swband = i
		}

		if (i > swband+1) && (mxaapq < math.Sqrt(float64(n))*tol) && (float64(n)*mxaapq*mxsinj < tol) {
			goto label1994
		}

		if notrot >= emptsw {
			goto label1994
		}

	}
	//     end i=1:NSWEEP loop
	//
	// #:( Reaching this point means that the procedure has not converged.
	info = nsweep - 1
	goto label1995

label1994:
	;
	// #:) Reaching this point means numerical convergence after the i-th
	//     sweep.

	info = 0
	// #:) INFO = 0 confirms successful iterations.
label1995:
	;

	//     Sort the singular values and find how many are above
	//     the underflow threshold.
	n2 = 0
	n4 = 0
	for p = 1; p <= n-1; p++ {
		q = sva.Off(p-1).Iamax(n-p+1, 1) + p - 1
		if p != q {
			temp1 = sva.Get(p - 1)
			sva.Set(p-1, sva.Get(q-1))
			sva.Set(q-1, temp1)
			temp1 = work.Get(p - 1)
			work.Set(p-1, work.Get(q-1))
			work.Set(q-1, temp1)
			a.Off(0, q-1).Vector().Swap(m, a.Off(0, p-1).Vector(), 1, 1)
			if rsvec {
				v.Off(0, q-1).Vector().Swap(mvl, v.Off(0, p-1).Vector(), 1, 1)
			}
		}
		if sva.Get(p-1) != zero {
			n4 = n4 + 1
			if sva.Get(p-1)*skl > sfmin {
				n2 = n2 + 1
			}
		}
	}
	if sva.Get(n-1) != zero {
		n4 = n4 + 1
		if sva.Get(n-1)*skl > sfmin {
			n2 = n2 + 1
		}
	}

	//     Normalize the left singular vectors.
	if lsvec || uctol {
		for p = 1; p <= n2; p++ {
			a.Off(0, p-1).Vector().Scal(m, work.Get(p-1)/sva.Get(p-1), 1)
		}
	}

	//     Scale the product of Jacobi rotations (assemble the fast rotations).
	if rsvec {
		if applv {
			for p = 1; p <= n; p++ {
				v.Off(0, p-1).Vector().Scal(mvl, work.Get(p-1), 1)
			}
		} else {
			for p = 1; p <= n; p++ {
				temp1 = one / v.Off(0, p-1).Vector().Nrm2(mvl, 1)
				v.Off(0, p-1).Vector().Scal(mvl, temp1, 1)
			}
		}
	}

	//     Undo scaling, if necessary (and possible).
	if ((skl > one) && (sva.Get(0) < (big / skl))) || ((skl < one) && (sva.Get(max(n2, 1)-1) > (sfmin / skl))) {
		for p = 1; p <= n; p++ {
			sva.Set(p-1, skl*sva.Get(p-1))
		}
		skl = one
	}

	work.Set(0, skl)
	//     The singular values of A are SKL*SVA(1:N). If SKL.NE.ONE
	//     then some of the singular values may overflow or underflow and
	//     the spectrum is given in this factored representation.

	work.Set(1, float64(n4))
	//     N4 is the number of computed nonzero singular values of A.

	work.Set(2, float64(n2))
	//     N2 is the number of singular values of A greater than SFMIN.
	//     If N2<N, SVA(N2:N) contains ZEROS and/or denormalized numbers
	//     that may carry some information.

	work.Set(3, float64(i))
	//     i is the index of the last sweep before declaring convergence.

	work.Set(4, mxaapq)
	//     MXAAPQ is the largest absolute value of scaled pivots in the
	//     last sweep

	work.Set(5, mxsinj)
	//     MXSINJ is the largest absolute value of the sines of Jacobi angles
	//     in the last sweep

	return
}
