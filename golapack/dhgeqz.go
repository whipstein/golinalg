package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dhgeqz computes the eigenvalues of a real matrix pair (H,T),
// where H is an upper Hessenberg matrix and T is upper triangular,
// using the double-shift QZ method.
// Matrix pairs of this type are produced by the reduction to
// generalized upper Hessenberg form of a real matrix pair (A,B):
//
//    A = Q1*H*Z1**T,  B = Q1*T*Z1**T,
//
// as computed by DGGHRD.
//
// If JOB='S', then the Hessenberg-triangular pair (H,T) is
// also reduced to generalized Schur form,
//
//    H = Q*S*Z**T,  T = Q*P*Z**T,
//
// where Q and Z are orthogonal matrices, P is an upper triangular
// matrix, and S is a quasi-triangular matrix with 1-by-1 and 2-by-2
// diagonal blocks.
//
// The 1-by-1 blocks correspond to real eigenvalues of the matrix pair
// (H,T) and the 2-by-2 blocks correspond to complex conjugate pairs of
// eigenvalues.
//
// Additionally, the 2-by-2 upper triangular diagonal blocks of P
// corresponding to 2-by-2 blocks of S are reduced to positive diagonal
// form, i.e., if S(j+1,j) is non-zero, then P(j+1,j) = P(j,j+1) = 0,
// P(j,j) > 0, and P(j+1,j+1) > 0.
//
// Optionally, the orthogonal matrix Q from the generalized Schur
// factorization may be postmultiplied into an input matrix Q1, and the
// orthogonal matrix Z may be postmultiplied into an input matrix Z1.
// If Q1 and Z1 are the orthogonal matrices from DGGHRD that reduced
// the matrix pair (A,B) to generalized upper Hessenberg form, then the
// output matrices Q1*Q and Z1*Z are the orthogonal factors from the
// generalized Schur factorization of (A,B):
//
//    A = (Q1*Q)*S*(Z1*Z)**T,  B = (Q1*Q)*P*(Z1*Z)**T.
//
// To avoid overflow, eigenvalues of the matrix pair (H,T) (equivalently,
// of (A,B)) are computed as a pair of values (alpha,beta), where alpha is
// complex and beta real.
// If beta is nonzero, lambda = alpha / beta is an eigenvalue of the
// generalized nonsymmetric eigenvalue problem (GNEP)
//    A*x = lambda*B*x
// and if alpha is nonzero, mu = beta / alpha is an eigenvalue of the
// alternate form of the GNEP
//    mu*A*y = B*y.
// Real eigenvalues can be read directly from the generalized Schur
// form:
//   alpha = S(i,i), beta = P(i,i).
//
// Ref: C.B. Moler & G.W. Stewart, "An Algorithm for Generalized Matrix
//      Eigenvalue Problems", SIAM J. Numer. Anal., 10(1973),
//      pp. 241--256.
func Dhgeqz(job, compq, compz byte, n, ilo, ihi *int, h *mat.Matrix, ldh *int, t *mat.Matrix, ldt *int, alphar, alphai, beta *mat.Vector, q *mat.Matrix, ldq *int, z *mat.Matrix, ldz *int, work *mat.Vector, lwork, info *int) {
	var ilazr2, ilazro, ilpivt, ilq, ilschr, ilz, lquery bool
	var a11, a12, a1i, a1r, a21, a22, a2i, a2r, ad11, ad11l, ad12, ad12l, ad21, ad21l, ad22, ad22l, ad32l, an, anorm, ascale, atol, b11, b1a, b1i, b1r, b22, b2a, b2i, b2r, bn, bnorm, bscale, btol, c, c11i, c11r, c12, c21, c22i, c22r, cl, cq, cr, cz, eshift, half, one, s, s1, s1inv, s2, safety, safmax, safmin, scale, sl, sqi, sqr, sr, szi, szr, t1, tau, temp, temp2, tempi, tempr, u1, u12, u12l, u2, ulp, vs, w11, w12, w21, w22, wabs, wi, wr, wr2, zero float64
	var icompq, icompz, ifirst, ifrstm, iiter, ilast, ilastm, in, ischur, istart, j, jc, jch, jiter, jr, maxit int

	v := vf(3)

	//    $                     SAFETY = 1.0E+0 )
	half = 0.5
	zero = 0.0
	one = 1.0
	safety = 1.0e+2

	//     Decode JOB, COMPQ, COMPZ
	if job == 'E' {
		ilschr = false
		ischur = 1
	} else if job == 'S' {
		ilschr = true
		ischur = 2
	} else {
		ischur = 0
	}

	if compq == 'N' {
		ilq = false
		icompq = 1
	} else if compq == 'V' {
		ilq = true
		icompq = 2
	} else if compq == 'I' {
		ilq = true
		icompq = 3
	} else {
		icompq = 0
	}

	if compz == 'N' {
		ilz = false
		icompz = 1
	} else if compz == 'V' {
		ilz = true
		icompz = 2
	} else if compz == 'I' {
		ilz = true
		icompz = 3
	} else {
		icompz = 0
	}

	//     Check Argument Values
	(*info) = 0
	work.Set(0, float64(maxint(1, *n)))
	lquery = ((*lwork) == -1)
	if ischur == 0 {
		(*info) = -1
	} else if icompq == 0 {
		(*info) = -2
	} else if icompz == 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ilo) < 1 {
		(*info) = -5
	} else if (*ihi) > (*n) || (*ihi) < (*ilo)-1 {
		(*info) = -6
	} else if (*ldh) < (*n) {
		(*info) = -8
	} else if (*ldt) < (*n) {
		(*info) = -10
	} else if (*ldq) < 1 || (ilq && (*ldq) < (*n)) {
		(*info) = -15
	} else if (*ldz) < 1 || (ilz && (*ldz) < (*n)) {
		(*info) = -17
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -19
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DHGEQZ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) <= 0 {
		work.Set(0, float64(int(1)))
		return
	}

	//     Initialize Q and Z
	if icompq == 3 {
		Dlaset('F', n, n, &zero, &one, q, ldq)
	}
	if icompz == 3 {
		Dlaset('F', n, n, &zero, &one, z, ldz)
	}

	//     Machine Constants
	in = (*ihi) + 1 - (*ilo)
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	ulp = Dlamch(Epsilon) * Dlamch(Base)
	anorm = Dlanhs('F', &in, h.Off((*ilo)-1, (*ilo)-1), ldh, work)
	bnorm = Dlanhs('F', &in, t.Off((*ilo)-1, (*ilo)-1), ldt, work)
	atol = maxf64(safmin, ulp*anorm)
	btol = maxf64(safmin, ulp*bnorm)
	ascale = one / maxf64(safmin, anorm)
	bscale = one / maxf64(safmin, bnorm)

	//     Set Eigenvalues IHI+1:N
	for j = (*ihi) + 1; j <= (*n); j++ {
		if t.Get(j-1, j-1) < zero {
			if ilschr {
				for jr = 1; jr <= j; jr++ {
					h.Set(jr-1, j-1, -h.Get(jr-1, j-1))
					t.Set(jr-1, j-1, -t.Get(jr-1, j-1))
				}
			} else {
				h.Set(j-1, j-1, -h.Get(j-1, j-1))
				t.Set(j-1, j-1, -t.Get(j-1, j-1))
			}
			if ilz {
				for jr = 1; jr <= (*n); jr++ {
					z.Set(jr-1, j-1, -z.Get(jr-1, j-1))
				}
			}
		}
		alphar.Set(j-1, h.Get(j-1, j-1))
		alphai.Set(j-1, zero)
		beta.Set(j-1, t.Get(j-1, j-1))
	}

	//     If IHI < ILO, skip QZ steps
	if (*ihi) < (*ilo) {
		goto label380
	}

	//     MAIN QZ ITERATION LOOP
	//
	//     Initialize dynamic indices
	//
	//     Eigenvalues ILAST+1:N have been found.
	//        Column operations modify rows IFRSTM:whatever.
	//        Row operations modify columns whatever:ILASTM.
	//
	//     If only eigenvalues are being computed, then
	//        IFRSTM is the row of the last splitting row above row ILAST;
	//        this is always at least ILO.
	//     IITER counts iterations since the last eigenvalue was found,
	//        to tell when to use an extraordinary shift.
	//     MAXIT is the maximum number of QZ sweeps allowed.
	ilast = (*ihi)
	if ilschr {
		ifrstm = 1
		ilastm = (*n)
	} else {
		ifrstm = (*ilo)
		ilastm = (*ihi)
	}
	iiter = 0
	eshift = zero
	maxit = 30 * ((*ihi) - (*ilo) + 1)

	for jiter = 1; jiter <= maxit; jiter++ {
		//        Split the matrix if possible.
		//
		//        Two tests:
		//           1: H(j,j-1)=0  or  j=ILO
		//           2: T(j,j)=0
		if ilast == (*ilo) {
			//           Special case: j=ILAST
			goto label80
		} else {
			if math.Abs(h.Get(ilast-1, ilast-1-1)) <= atol {
				h.Set(ilast-1, ilast-1-1, zero)
				goto label80
			}
		}

		if math.Abs(t.Get(ilast-1, ilast-1)) <= btol {
			t.Set(ilast-1, ilast-1, zero)
			goto label70
		}

		//        General case: j<ILAST
		for j = ilast - 1; j >= (*ilo); j-- {
			//           Test 1: for H(j,j-1)=0 or j=ILO
			if j == (*ilo) {
				ilazro = true
			} else {
				if math.Abs(h.Get(j-1, j-1-1)) <= atol {
					h.Set(j-1, j-1-1, zero)
					ilazro = true
				} else {
					ilazro = false
				}
			}

			//           Test 2: for T(j,j)=0
			if math.Abs(t.Get(j-1, j-1)) < btol {
				t.Set(j-1, j-1, zero)

				//              Test 1a: Check for 2 consecutive small subdiagonals in A
				ilazr2 = false
				if !ilazro {
					temp = math.Abs(h.Get(j-1, j-1-1))
					temp2 = math.Abs(h.Get(j-1, j-1))
					tempr = maxf64(temp, temp2)
					if tempr < one && tempr != zero {
						temp = temp / tempr
						temp2 = temp2 / tempr
					}
					if temp*(ascale*math.Abs(h.Get(j+1-1, j-1))) <= temp2*(ascale*atol) {
						ilazr2 = true
					}
				}

				//              If both tests pass (1 & 2), i.e., the leading diagonal
				//              element of B in the block is zero, split a 1x1 block off
				//              at the top. (I.e., at the J-th row/column) The leading
				//              diagonal element of the remainder can also be zero, so
				//              this may have to be done repeatedly.
				if ilazro || ilazr2 {
					for jch = j; jch <= ilast-1; jch++ {
						temp = h.Get(jch-1, jch-1)
						Dlartg(&temp, h.GetPtr(jch+1-1, jch-1), &c, &s, h.GetPtr(jch-1, jch-1))
						h.Set(jch+1-1, jch-1, zero)
						goblas.Drot(ilastm-jch, h.Vector(jch-1, jch+1-1), *ldh, h.Vector(jch+1-1, jch+1-1), *ldh, c, s)
						goblas.Drot(ilastm-jch, t.Vector(jch-1, jch+1-1), *ldt, t.Vector(jch+1-1, jch+1-1), *ldt, c, s)
						if ilq {
							goblas.Drot(*n, q.Vector(0, jch-1), 1, q.Vector(0, jch+1-1), 1, c, s)
						}
						if ilazr2 {
							h.Set(jch-1, jch-1-1, h.Get(jch-1, jch-1-1)*c)
						}
						ilazr2 = false
						if math.Abs(t.Get(jch+1-1, jch+1-1)) >= btol {
							if jch+1 >= ilast {
								goto label80
							} else {
								ifirst = jch + 1
								goto label110
							}
						}
						t.Set(jch+1-1, jch+1-1, zero)
					}
					goto label70
				} else {
					//                 Only test 2 passed -- chase the zero to T(ILAST,ILAST)
					//                 Then process as in the case T(ILAST,ILAST)=0
					for jch = j; jch <= ilast-1; jch++ {
						temp = t.Get(jch-1, jch+1-1)
						Dlartg(&temp, t.GetPtr(jch+1-1, jch+1-1), &c, &s, t.GetPtr(jch-1, jch+1-1))
						t.Set(jch+1-1, jch+1-1, zero)
						if jch < ilastm-1 {
							goblas.Drot(ilastm-jch-1, t.Vector(jch-1, jch+2-1), *ldt, t.Vector(jch+1-1, jch+2-1), *ldt, c, s)
						}
						goblas.Drot(ilastm-jch+2, h.Vector(jch-1, jch-1-1), *ldh, h.Vector(jch+1-1, jch-1-1), *ldh, c, s)
						if ilq {
							goblas.Drot(*n, q.Vector(0, jch-1), 1, q.Vector(0, jch+1-1), 1, c, s)
						}
						temp = h.Get(jch+1-1, jch-1)
						Dlartg(&temp, h.GetPtr(jch+1-1, jch-1-1), &c, &s, h.GetPtr(jch+1-1, jch-1))
						h.Set(jch+1-1, jch-1-1, zero)
						goblas.Drot(jch+1-ifrstm, h.Vector(ifrstm-1, jch-1), 1, h.Vector(ifrstm-1, jch-1-1), 1, c, s)
						goblas.Drot(jch-ifrstm, t.Vector(ifrstm-1, jch-1), 1, t.Vector(ifrstm-1, jch-1-1), 1, c, s)
						if ilz {
							goblas.Drot(*n, z.Vector(0, jch-1), 1, z.Vector(0, jch-1-1), 1, c, s)
						}
					}
					goto label70
				}
			} else if ilazro {
				//              Only test 1 passed -- work on J:ILAST
				ifirst = j
				goto label110
			}

			//           Neither test passed -- try next J
		}
		//        (Drop-through is "impossible")
		(*info) = (*n) + 1
		goto label420

		//        T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
		//        1x1 block.
	label70:
		;
		temp = h.Get(ilast-1, ilast-1)
		Dlartg(&temp, h.GetPtr(ilast-1, ilast-1-1), &c, &s, h.GetPtr(ilast-1, ilast-1))
		h.Set(ilast-1, ilast-1-1, zero)
		goblas.Drot(ilast-ifrstm, h.Vector(ifrstm-1, ilast-1), 1, h.Vector(ifrstm-1, ilast-1-1), 1, c, s)
		goblas.Drot(ilast-ifrstm, t.Vector(ifrstm-1, ilast-1), 1, t.Vector(ifrstm-1, ilast-1-1), 1, c, s)
		if ilz {
			goblas.Drot(*n, z.Vector(0, ilast-1), 1, z.Vector(0, ilast-1-1), 1, c, s)
		}

		//        H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHAR, ALPHAI,
		//                              and BETA
	label80:
		;
		if t.Get(ilast-1, ilast-1) < zero {
			if ilschr {
				for j = ifrstm; j <= ilast; j++ {
					h.Set(j-1, ilast-1, -h.Get(j-1, ilast-1))
					t.Set(j-1, ilast-1, -t.Get(j-1, ilast-1))
				}
			} else {
				h.Set(ilast-1, ilast-1, -h.Get(ilast-1, ilast-1))
				t.Set(ilast-1, ilast-1, -t.Get(ilast-1, ilast-1))
			}
			if ilz {
				for j = 1; j <= (*n); j++ {
					z.Set(j-1, ilast-1, -z.Get(j-1, ilast-1))
				}
			}
		}
		alphar.Set(ilast-1, h.Get(ilast-1, ilast-1))
		alphai.Set(ilast-1, zero)
		beta.Set(ilast-1, t.Get(ilast-1, ilast-1))

		//        Go to next block -- exit if finished.
		ilast = ilast - 1
		if ilast < (*ilo) {
			goto label380
		}

		//        Reset counters
		iiter = 0
		eshift = zero
		if !ilschr {
			ilastm = ilast
			if ifrstm > ilast {
				ifrstm = (*ilo)
			}
		}
		goto label350

		//        QZ step
		//
		//        This iteration only involves rows/columns IFIRST:ILAST. We
		//        assume IFIRST < ILAST, and that the diagonal of B is non-zero.
	label110:
		;
		iiter = iiter + 1
		if !ilschr {
			ifrstm = ifirst
		}

		//        Compute single shifts.
		//
		//        At this point, IFIRST < ILAST, and the diagonal elements of
		//        T(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in
		//        magnitude)
		if (iiter/10)*10 == iiter {
			//           Exceptional shift.  Chosen for no particularly good reason.
			//           (Single shift only.)
			if (float64(maxit)*safmin)*math.Abs(h.Get(ilast-1, ilast-1-1)) < math.Abs(t.Get(ilast-1-1, ilast-1-1)) {
				eshift = h.Get(ilast-1, ilast-1-1) / t.Get(ilast-1-1, ilast-1-1)
			} else {
				eshift = eshift + one/(safmin*float64(maxit))
			}
			s1 = one
			wr = eshift

		} else {
			//           Shifts based on the generalized eigenvalues of the
			//           bottom-right 2x2 block of A and B. The first eigenvalue
			//           returned by DLAG2 is the Wilkinson shift (AEP p.512),
			Dlag2(h.Off(ilast-1-1, ilast-1-1), ldh, t.Off(ilast-1-1, ilast-1-1), ldt, toPtrf64(safmin*safety), &s1, &s2, &wr, &wr2, &wi)

			if math.Abs((wr/s1)*t.Get(ilast-1, ilast-1)-h.Get(ilast-1, ilast-1)) > math.Abs((wr2/s2)*t.Get(ilast-1, ilast-1)-h.Get(ilast-1, ilast-1)) {
				temp = wr
				wr = wr2
				wr2 = temp
				temp = s1
				s1 = s2
				s2 = temp
			}
			temp = maxf64(s1, safmin*maxf64(one, math.Abs(wr), math.Abs(wi)))
			if wi != zero {
				goto label200
			}
		}

		//        Fiddle with shift to avoid overflow
		temp = minf64(ascale, one) * (half * safmax)
		if s1 > temp {
			scale = temp / s1
		} else {
			scale = one
		}

		temp = minf64(bscale, one) * (half * safmax)
		if math.Abs(wr) > temp {
			scale = minf64(scale, temp/math.Abs(wr))
		}
		s1 = scale * s1
		wr = scale * wr

		//        Now check for two consecutive small subdiagonals.
		for j = ilast - 1; j >= ifirst+1; j-- {
			istart = j
			temp = math.Abs(s1 * h.Get(j-1, j-1-1))
			temp2 = math.Abs(s1*h.Get(j-1, j-1) - wr*t.Get(j-1, j-1))
			tempr = maxf64(temp, temp2)
			if tempr < one && tempr != zero {
				temp = temp / tempr
				temp2 = temp2 / tempr
			}
			if math.Abs((ascale*h.Get(j+1-1, j-1))*temp) <= (ascale*atol)*temp2 {
				goto label130
			}
		}

		istart = ifirst
	label130:
		;

		//        Do an implicit single-shift QZ sweep.
		//
		//        Initial Q
		temp = s1*h.Get(istart-1, istart-1) - wr*t.Get(istart-1, istart-1)
		temp2 = s1 * h.Get(istart+1-1, istart-1)
		Dlartg(&temp, &temp2, &c, &s, &tempr)

		//        Sweep
		for j = istart; j <= ilast-1; j++ {
			if j > istart {
				temp = h.Get(j-1, j-1-1)
				Dlartg(&temp, h.GetPtr(j+1-1, j-1-1), &c, &s, h.GetPtr(j-1, j-1-1))
				h.Set(j+1-1, j-1-1, zero)
			}

			for jc = j; jc <= ilastm; jc++ {
				temp = c*h.Get(j-1, jc-1) + s*h.Get(j+1-1, jc-1)
				h.Set(j+1-1, jc-1, -s*h.Get(j-1, jc-1)+c*h.Get(j+1-1, jc-1))
				h.Set(j-1, jc-1, temp)
				temp2 = c*t.Get(j-1, jc-1) + s*t.Get(j+1-1, jc-1)
				t.Set(j+1-1, jc-1, -s*t.Get(j-1, jc-1)+c*t.Get(j+1-1, jc-1))
				t.Set(j-1, jc-1, temp2)
			}
			if ilq {
				for jr = 1; jr <= (*n); jr++ {
					temp = c*q.Get(jr-1, j-1) + s*q.Get(jr-1, j+1-1)
					q.Set(jr-1, j+1-1, -s*q.Get(jr-1, j-1)+c*q.Get(jr-1, j+1-1))
					q.Set(jr-1, j-1, temp)
				}
			}

			temp = t.Get(j+1-1, j+1-1)
			Dlartg(&temp, t.GetPtr(j+1-1, j-1), &c, &s, t.GetPtr(j+1-1, j+1-1))
			t.Set(j+1-1, j-1, zero)

			for jr = ifrstm; jr <= minint(j+2, ilast); jr++ {
				temp = c*h.Get(jr-1, j+1-1) + s*h.Get(jr-1, j-1)
				h.Set(jr-1, j-1, -s*h.Get(jr-1, j+1-1)+c*h.Get(jr-1, j-1))
				h.Set(jr-1, j+1-1, temp)
			}
			for jr = ifrstm; jr <= j; jr++ {
				temp = c*t.Get(jr-1, j+1-1) + s*t.Get(jr-1, j-1)
				t.Set(jr-1, j-1, -s*t.Get(jr-1, j+1-1)+c*t.Get(jr-1, j-1))
				t.Set(jr-1, j+1-1, temp)
			}
			if ilz {
				for jr = 1; jr <= (*n); jr++ {
					temp = c*z.Get(jr-1, j+1-1) + s*z.Get(jr-1, j-1)
					z.Set(jr-1, j-1, -s*z.Get(jr-1, j+1-1)+c*z.Get(jr-1, j-1))
					z.Set(jr-1, j+1-1, temp)
				}
			}
		}

		goto label350

		//        Use Francis double-shift
		//
		//        Note: the Francis double-shift should work with real shifts,
		//              but only if the block is at least 3x3.
		//              This code may break if this point is reached with
		//              a 2x2 block with real eigenvalues.
	label200:
		;
		if ifirst+1 == ilast {
			//           Special case -- 2x2 block with complex eigenvectors
			//
			//           Step 1: Standardize, that is, rotate so that
			//
			//                       ( B11  0  )
			//                   B = (         )  with B11 non-negative.
			//                       (  0  B22 )
			Dlasv2(t.GetPtr(ilast-1-1, ilast-1-1), t.GetPtr(ilast-1-1, ilast-1), t.GetPtr(ilast-1, ilast-1), &b22, &b11, &sr, &cr, &sl, &cl)

			if b11 < zero {
				cr = -cr
				sr = -sr
				b11 = -b11
				b22 = -b22
			}

			goblas.Drot(ilastm+1-ifirst, h.Vector(ilast-1-1, ilast-1-1), *ldh, h.Vector(ilast-1, ilast-1-1), *ldh, cl, sl)
			goblas.Drot(ilast+1-ifrstm, h.Vector(ifrstm-1, ilast-1-1), 1, h.Vector(ifrstm-1, ilast-1), 1, cr, sr)

			if ilast < ilastm {
				goblas.Drot(ilastm-ilast, t.Vector(ilast-1-1, ilast+1-1), *ldt, t.Vector(ilast-1, ilast+1-1), *ldt, cl, sl)
			}
			if ifrstm < ilast-1 {
				goblas.Drot(ifirst-ifrstm, t.Vector(ifrstm-1, ilast-1-1), 1, t.Vector(ifrstm-1, ilast-1), 1, cr, sr)
			}

			if ilq {
				goblas.Drot(*n, q.Vector(0, ilast-1-1), 1, q.Vector(0, ilast-1), 1, cl, sl)
			}
			if ilz {
				goblas.Drot(*n, z.Vector(0, ilast-1-1), 1, z.Vector(0, ilast-1), 1, cr, sr)
			}

			t.Set(ilast-1-1, ilast-1-1, b11)
			t.Set(ilast-1-1, ilast-1, zero)
			t.Set(ilast-1, ilast-1-1, zero)
			t.Set(ilast-1, ilast-1, b22)

			//           If B22 is negative, negate column ILAST
			if b22 < zero {
				for j = ifrstm; j <= ilast; j++ {
					h.Set(j-1, ilast-1, -h.Get(j-1, ilast-1))
					t.Set(j-1, ilast-1, -t.Get(j-1, ilast-1))
				}

				if ilz {
					for j = 1; j <= (*n); j++ {
						z.Set(j-1, ilast-1, -z.Get(j-1, ilast-1))
					}
				}
				b22 = -b22
			}

			//           Step 2: Compute ALPHAR, ALPHAI, and BETA (see refs.)
			//
			//           Recompute shift
			Dlag2(h.Off(ilast-1-1, ilast-1-1), ldh, t.Off(ilast-1-1, ilast-1-1), ldt, toPtrf64(safmin*safety), &s1, &temp, &wr, &temp2, &wi)

			//           If standardization has perturbed the shift onto real line,
			//           do another (real single-shift) QR step.
			if wi == zero {
				goto label350
			}
			s1inv = one / s1

			//           Do EISPACK (QZVAL) computation of alpha and beta
			a11 = h.Get(ilast-1-1, ilast-1-1)
			a21 = h.Get(ilast-1, ilast-1-1)
			a12 = h.Get(ilast-1-1, ilast-1)
			a22 = h.Get(ilast-1, ilast-1)

			//           Compute complex Givens rotation on right
			//           (Assume some element of C = (sA - wB) > unfl )
			//                            __
			//           (sA - wB) ( CZ   -SZ )
			//                     ( SZ    CZ )
			c11r = s1*a11 - wr*b11
			c11i = -wi * b11
			c12 = s1 * a12
			c21 = s1 * a21
			c22r = s1*a22 - wr*b22
			c22i = -wi * b22

			if math.Abs(c11r)+math.Abs(c11i)+math.Abs(c12) > math.Abs(c21)+math.Abs(c22r)+math.Abs(c22i) {
				t1 = Dlapy3(&c12, &c11r, &c11i)
				cz = c12 / t1
				szr = -c11r / t1
				szi = -c11i / t1
			} else {
				cz = Dlapy2(&c22r, &c22i)
				if cz <= safmin {
					cz = zero
					szr = one
					szi = zero
				} else {
					tempr = c22r / cz
					tempi = c22i / cz
					t1 = Dlapy2(&cz, &c21)
					cz = cz / t1
					szr = -c21 * tempr / t1
					szi = c21 * tempi / t1
				}
			}

			//           Compute Givens rotation on left
			//
			//           (  CQ   SQ )
			//           (  __      )  A or B
			//           ( -SQ   CQ )
			an = math.Abs(a11) + math.Abs(a12) + math.Abs(a21) + math.Abs(a22)
			bn = math.Abs(b11) + math.Abs(b22)
			wabs = math.Abs(wr) + math.Abs(wi)
			if s1*an > wabs*bn {
				cq = cz * b11
				sqr = szr * b22
				sqi = -szi * b22
			} else {
				a1r = cz*a11 + szr*a12
				a1i = szi * a12
				a2r = cz*a21 + szr*a22
				a2i = szi * a22
				cq = Dlapy2(&a1r, &a1i)
				if cq <= safmin {
					cq = zero
					sqr = one
					sqi = zero
				} else {
					tempr = a1r / cq
					tempi = a1i / cq
					sqr = tempr*a2r + tempi*a2i
					sqi = tempi*a2r - tempr*a2i
				}
			}
			t1 = Dlapy3(&cq, &sqr, &sqi)
			cq = cq / t1
			sqr = sqr / t1
			sqi = sqi / t1

			//           Compute diagonal elements of QBZ
			tempr = sqr*szr - sqi*szi
			tempi = sqr*szi + sqi*szr
			b1r = cq*cz*b11 + tempr*b22
			b1i = tempi * b22
			b1a = Dlapy2(&b1r, &b1i)
			b2r = cq*cz*b22 + tempr*b11
			b2i = -tempi * b11
			b2a = Dlapy2(&b2r, &b2i)

			//           Normalize so beta > 0, and Im( alpha1 ) > 0
			beta.Set(ilast-1-1, b1a)
			beta.Set(ilast-1, b2a)
			alphar.Set(ilast-1-1, (wr*b1a)*s1inv)
			alphai.Set(ilast-1-1, (wi*b1a)*s1inv)
			alphar.Set(ilast-1, (wr*b2a)*s1inv)
			alphai.Set(ilast-1, -(wi*b2a)*s1inv)

			//           Step 3: Go to next block -- exit if finished.
			ilast = ifirst - 1
			if ilast < (*ilo) {
				goto label380
			}

			//           Reset counters
			iiter = 0
			eshift = zero
			if !ilschr {
				ilastm = ilast
				if ifrstm > ilast {
					ifrstm = (*ilo)
				}
			}
			goto label350
		} else {
			//           Usual case: 3x3 or larger block, using Francis implicit
			//                       double-shift
			//
			//                                    2
			//           Eigenvalue equation is  w  - c w + d = 0,
			//
			//                                         -1 2        -1
			//           so compute 1st column of  (A B  )  - c A B   + d
			//           using the formula in QZIT (from EISPACK)
			//
			//           We assume that the block is at least 3x3
			ad11 = (ascale * h.Get(ilast-1-1, ilast-1-1)) / (bscale * t.Get(ilast-1-1, ilast-1-1))
			ad21 = (ascale * h.Get(ilast-1, ilast-1-1)) / (bscale * t.Get(ilast-1-1, ilast-1-1))
			ad12 = (ascale * h.Get(ilast-1-1, ilast-1)) / (bscale * t.Get(ilast-1, ilast-1))
			ad22 = (ascale * h.Get(ilast-1, ilast-1)) / (bscale * t.Get(ilast-1, ilast-1))
			u12 = t.Get(ilast-1-1, ilast-1) / t.Get(ilast-1, ilast-1)
			ad11l = (ascale * h.Get(ifirst-1, ifirst-1)) / (bscale * t.Get(ifirst-1, ifirst-1))
			ad21l = (ascale * h.Get(ifirst+1-1, ifirst-1)) / (bscale * t.Get(ifirst-1, ifirst-1))
			ad12l = (ascale * h.Get(ifirst-1, ifirst+1-1)) / (bscale * t.Get(ifirst+1-1, ifirst+1-1))
			ad22l = (ascale * h.Get(ifirst+1-1, ifirst+1-1)) / (bscale * t.Get(ifirst+1-1, ifirst+1-1))
			ad32l = (ascale * h.Get(ifirst+2-1, ifirst+1-1)) / (bscale * t.Get(ifirst+1-1, ifirst+1-1))
			u12l = t.Get(ifirst-1, ifirst+1-1) / t.Get(ifirst+1-1, ifirst+1-1)

			v.Set(0, (ad11-ad11l)*(ad22-ad11l)-ad12*ad21+ad21*u12*ad11l+(ad12l-ad11l*u12l)*ad21l)
			v.Set(1, ((ad22l-ad11l)-ad21l*u12l-(ad11-ad11l)-(ad22-ad11l)+ad21*u12)*ad21l)
			v.Set(2, ad32l*ad21l)

			istart = ifirst

			Dlarfg(func() *int { y := 3; return &y }(), v.GetPtr(0), v.Off(1), func() *int { y := 1; return &y }(), &tau)
			v.Set(0, one)

			//           Sweep
			for j = istart; j <= ilast-2; j++ {
				//              All but last elements: use 3x3 Householder transforms.
				//
				//              Zero (j-1)st column of A
				if j > istart {
					v.Set(0, h.Get(j-1, j-1-1))
					v.Set(1, h.Get(j+1-1, j-1-1))
					v.Set(2, h.Get(j+2-1, j-1-1))
					//
					Dlarfg(func() *int { y := 3; return &y }(), h.GetPtr(j-1, j-1-1), v.Off(1), func() *int { y := 1; return &y }(), &tau)
					v.Set(0, one)
					h.Set(j+1-1, j-1-1, zero)
					h.Set(j+2-1, j-1-1, zero)
				}

				for jc = j; jc <= ilastm; jc++ {
					temp = tau * (h.Get(j-1, jc-1) + v.Get(1)*h.Get(j+1-1, jc-1) + v.Get(2)*h.Get(j+2-1, jc-1))
					h.Set(j-1, jc-1, h.Get(j-1, jc-1)-temp)
					h.Set(j+1-1, jc-1, h.Get(j+1-1, jc-1)-temp*v.Get(1))
					h.Set(j+2-1, jc-1, h.Get(j+2-1, jc-1)-temp*v.Get(2))
					temp2 = tau * (t.Get(j-1, jc-1) + v.Get(1)*t.Get(j+1-1, jc-1) + v.Get(2)*t.Get(j+2-1, jc-1))
					t.Set(j-1, jc-1, t.Get(j-1, jc-1)-temp2)
					t.Set(j+1-1, jc-1, t.Get(j+1-1, jc-1)-temp2*v.Get(1))
					t.Set(j+2-1, jc-1, t.Get(j+2-1, jc-1)-temp2*v.Get(2))
				}
				if ilq {
					for jr = 1; jr <= (*n); jr++ {
						temp = tau * (q.Get(jr-1, j-1) + v.Get(1)*q.Get(jr-1, j+1-1) + v.Get(2)*q.Get(jr-1, j+2-1))
						q.Set(jr-1, j-1, q.Get(jr-1, j-1)-temp)
						q.Set(jr-1, j+1-1, q.Get(jr-1, j+1-1)-temp*v.Get(1))
						q.Set(jr-1, j+2-1, q.Get(jr-1, j+2-1)-temp*v.Get(2))
					}
				}

				//              Zero j-th column of B (see DLAGBC for details)
				//
				//              Swap rows to pivot
				ilpivt = false
				temp = maxf64(math.Abs(t.Get(j+1-1, j+1-1)), math.Abs(t.Get(j+1-1, j+2-1)))
				temp2 = maxf64(math.Abs(t.Get(j+2-1, j+1-1)), math.Abs(t.Get(j+2-1, j+2-1)))
				if maxf64(temp, temp2) < safmin {
					scale = zero
					u1 = one
					u2 = zero
					goto label250
				} else if temp >= temp2 {
					w11 = t.Get(j+1-1, j+1-1)
					w21 = t.Get(j+2-1, j+1-1)
					w12 = t.Get(j+1-1, j+2-1)
					w22 = t.Get(j+2-1, j+2-1)
					u1 = t.Get(j+1-1, j-1)
					u2 = t.Get(j+2-1, j-1)
				} else {
					w21 = t.Get(j+1-1, j+1-1)
					w11 = t.Get(j+2-1, j+1-1)
					w22 = t.Get(j+1-1, j+2-1)
					w12 = t.Get(j+2-1, j+2-1)
					u2 = t.Get(j+1-1, j-1)
					u1 = t.Get(j+2-1, j-1)
				}

				//              Swap columns if nec.
				if math.Abs(w12) > math.Abs(w11) {
					ilpivt = true
					temp = w12
					temp2 = w22
					w12 = w11
					w22 = w21
					w11 = temp
					w21 = temp2
				}

				//              LU-factor
				temp = w21 / w11
				u2 = u2 - temp*u1
				w22 = w22 - temp*w12
				w21 = zero

				//              Compute SCALE
				scale = one
				if math.Abs(w22) < safmin {
					scale = zero
					u2 = one
					u1 = -w12 / w11
					goto label250
				}
				if math.Abs(w22) < math.Abs(u2) {
					scale = math.Abs(w22 / u2)
				}
				if math.Abs(w11) < math.Abs(u1) {
					scale = minf64(scale, math.Abs(w11/u1))
				}

				//              Solve
				u2 = (scale * u2) / w22
				u1 = (scale*u1 - w12*u2) / w11

			label250:
				;
				if ilpivt {
					temp = u2
					u2 = u1
					u1 = temp
				}

				//              Compute Householder Vector
				t1 = math.Sqrt(math.Pow(scale, 2) + math.Pow(u1, 2) + math.Pow(u2, 2))
				tau = one + scale/t1
				vs = -one / (scale + t1)
				v.Set(0, one)
				v.Set(1, vs*u1)
				v.Set(2, vs*u2)

				//              Apply transformations from the right.
				for jr = ifrstm; jr <= minint(j+3, ilast); jr++ {
					temp = tau * (h.Get(jr-1, j-1) + v.Get(1)*h.Get(jr-1, j+1-1) + v.Get(2)*h.Get(jr-1, j+2-1))
					h.Set(jr-1, j-1, h.Get(jr-1, j-1)-temp)
					h.Set(jr-1, j+1-1, h.Get(jr-1, j+1-1)-temp*v.Get(1))
					h.Set(jr-1, j+2-1, h.Get(jr-1, j+2-1)-temp*v.Get(2))
				}
				for jr = ifrstm; jr <= j+2; jr++ {
					temp = tau * (t.Get(jr-1, j-1) + v.Get(1)*t.Get(jr-1, j+1-1) + v.Get(2)*t.Get(jr-1, j+2-1))
					t.Set(jr-1, j-1, t.Get(jr-1, j-1)-temp)
					t.Set(jr-1, j+1-1, t.Get(jr-1, j+1-1)-temp*v.Get(1))
					t.Set(jr-1, j+2-1, t.Get(jr-1, j+2-1)-temp*v.Get(2))
				}
				if ilz {
					for jr = 1; jr <= (*n); jr++ {
						temp = tau * (z.Get(jr-1, j-1) + v.Get(1)*z.Get(jr-1, j+1-1) + v.Get(2)*z.Get(jr-1, j+2-1))
						z.Set(jr-1, j-1, z.Get(jr-1, j-1)-temp)
						z.Set(jr-1, j+1-1, z.Get(jr-1, j+1-1)-temp*v.Get(1))
						z.Set(jr-1, j+2-1, z.Get(jr-1, j+2-1)-temp*v.Get(2))
					}
				}
				t.Set(j+1-1, j-1, zero)
				t.Set(j+2-1, j-1, zero)
			}

			//           Last elements: Use Givens rotations
			//
			//           Rotations from the left
			j = ilast - 1
			temp = h.Get(j-1, j-1-1)
			Dlartg(&temp, h.GetPtr(j+1-1, j-1-1), &c, &s, h.GetPtr(j-1, j-1-1))
			h.Set(j+1-1, j-1-1, zero)

			for jc = j; jc <= ilastm; jc++ {
				temp = c*h.Get(j-1, jc-1) + s*h.Get(j+1-1, jc-1)
				h.Set(j+1-1, jc-1, -s*h.Get(j-1, jc-1)+c*h.Get(j+1-1, jc-1))
				h.Set(j-1, jc-1, temp)
				temp2 = c*t.Get(j-1, jc-1) + s*t.Get(j+1-1, jc-1)
				t.Set(j+1-1, jc-1, -s*t.Get(j-1, jc-1)+c*t.Get(j+1-1, jc-1))
				t.Set(j-1, jc-1, temp2)
			}
			if ilq {
				for jr = 1; jr <= (*n); jr++ {
					temp = c*q.Get(jr-1, j-1) + s*q.Get(jr-1, j+1-1)
					q.Set(jr-1, j+1-1, -s*q.Get(jr-1, j-1)+c*q.Get(jr-1, j+1-1))
					q.Set(jr-1, j-1, temp)
				}
			}

			//           Rotations from the right.
			temp = t.Get(j+1-1, j+1-1)
			Dlartg(&temp, t.GetPtr(j+1-1, j-1), &c, &s, t.GetPtr(j+1-1, j+1-1))
			t.Set(j+1-1, j-1, zero)

			for jr = ifrstm; jr <= ilast; jr++ {
				temp = c*h.Get(jr-1, j+1-1) + s*h.Get(jr-1, j-1)
				h.Set(jr-1, j-1, -s*h.Get(jr-1, j+1-1)+c*h.Get(jr-1, j-1))
				h.Set(jr-1, j+1-1, temp)
			}
			for jr = ifrstm; jr <= ilast-1; jr++ {
				temp = c*t.Get(jr-1, j+1-1) + s*t.Get(jr-1, j-1)
				t.Set(jr-1, j-1, -s*t.Get(jr-1, j+1-1)+c*t.Get(jr-1, j-1))
				t.Set(jr-1, j+1-1, temp)
			}
			if ilz {
				for jr = 1; jr <= (*n); jr++ {
					temp = c*z.Get(jr-1, j+1-1) + s*z.Get(jr-1, j-1)
					z.Set(jr-1, j-1, -s*z.Get(jr-1, j+1-1)+c*z.Get(jr-1, j-1))
					z.Set(jr-1, j+1-1, temp)
				}
			}

			//           End of Double-Shift code
		}

		goto label350

		//        End of iteration loop
	label350:
	}

	//     Drop-through = non-convergence
	(*info) = ilast
	goto label420

	//     Successful completion of all QZ steps
label380:
	;

	//     Set Eigenvalues 1:ILO-1
	for j = 1; j <= (*ilo)-1; j++ {
		if t.Get(j-1, j-1) < zero {
			if ilschr {
				for jr = 1; jr <= j; jr++ {
					h.Set(jr-1, j-1, -h.Get(jr-1, j-1))
					t.Set(jr-1, j-1, -t.Get(jr-1, j-1))
				}
			} else {
				h.Set(j-1, j-1, -h.Get(j-1, j-1))
				t.Set(j-1, j-1, -t.Get(j-1, j-1))
			}
			if ilz {
				for jr = 1; jr <= (*n); jr++ {
					z.Set(jr-1, j-1, -z.Get(jr-1, j-1))
				}
			}
		}
		alphar.Set(j-1, h.Get(j-1, j-1))
		alphai.Set(j-1, zero)
		beta.Set(j-1, t.Get(j-1, j-1))
	}

	//     Normal Termination
	(*info) = 0

	//     Exit (other than argument error) -- return optimal workspace size
label420:
	;
	work.Set(0, float64(*n))
}
