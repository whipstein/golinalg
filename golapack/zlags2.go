package golapack

import (
	"math"
	"math/cmplx"
)

// Zlags2 computes 2-by-2 unitary matrices U, V and Q, such
// that if ( UPPER ) then
//
//           U**H *A*Q = U**H *( A1 A2 )*Q = ( x  0  )
//                             ( 0  A3 )     ( x  x  )
// and
//           V**H*B*Q = V**H *( B1 B2 )*Q = ( x  0  )
//                            ( 0  B3 )     ( x  x  )
//
// or if ( .NOT.UPPER ) then
//
//           U**H *A*Q = U**H *( A1 0  )*Q = ( x  x  )
//                             ( A2 A3 )     ( 0  x  )
// and
//           V**H *B*Q = V**H *( B1 0  )*Q = ( x  x  )
//                             ( B2 B3 )     ( 0  x  )
// where
//
//   U = (   CSU    SNU ), V = (  CSV    SNV ),
//       ( -SNU**H  CSU )      ( -SNV**H CSV )
//
//   Q = (   CSQ    SNQ )
//       ( -SNQ**H  CSQ )
//
// The rows of the transformed A and B are parallel. Moreover, if the
// input 2-by-2 matrix A is not zero, then the transformed (1,1) entry
// of A is not zero. If the input matrices A and B are both not zero,
// then the transformed (2,2) element of B is not zero, except when the
// first rows of input A and B are parallel and the second rows are
// zero.
func Zlags2(upper bool, a1 float64, a2 complex128, a3, b1 float64, b2 complex128, b3 float64) (csu float64, snu complex128, csv float64, snv complex128, csq float64, snq complex128) {
	var b, c, d1, ua11, ua12, ua21, ua22, vb11, vb12, vb21, vb22 complex128
	var a, aua11, aua12, aua21, aua22, avb11, avb12, avb21, avb22, csl, csr, d, fb, fc, one, snl, snr, ua11r, ua22r, vb11r, vb22r, zero float64

	zero = 0.0
	one = 1.0

	if upper {
		//        Input matrices A and B are upper triangular matrices
		//
		//        Form matrix C = A*adj(B) = ( a b )
		//                                   ( 0 d )
		a = a1 * b3
		d = a3 * b1
		b = a2*toCmplx(b1) - toCmplx(a1)*b2
		fb = cmplx.Abs(b)

		//        Transform complex 2-by-2 matrix C to real matrix by unitary
		//        diagonal matrix diag(1,D1).
		d1 = toCmplx(one)
		if fb != zero {
			d1 = b / complex(fb, 0)
		}

		//        The SVD of real 2 by 2 triangular C
		//
		//         ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 )
		//         ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T )
		_, _, snr, csr, snl, csl = Dlasv2(a, fb, d)

		if math.Abs(csl) >= math.Abs(snl) || math.Abs(csr) >= math.Abs(snr) {
			//           Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
			//           and (1,2) element of |U|**H *|A| and |V|**H *|B|.
			ua11r = csl * a1
			ua12 = toCmplx(csl)*a2 + d1*toCmplx(snl*a3)

			vb11r = csr * b1
			vb12 = toCmplx(csr)*b2 + d1*toCmplx(snr*b3)

			aua12 = math.Abs(csl)*abs1(a2) + math.Abs(snl)*math.Abs(a3)
			avb12 = math.Abs(csr)*abs1(b2) + math.Abs(snr)*math.Abs(b3)

			//           zero (1,2) elements of U**H *A and V**H *B
			if (math.Abs(ua11r) + abs1(ua12)) == zero {
				csq, snq, _ = Zlartg(toCmplx(-vb11r), cmplx.Conj(vb12))
			} else if (math.Abs(vb11r) + abs1(vb12)) == zero {
				csq, snq, _ = Zlartg(toCmplx(-ua11r), cmplx.Conj(ua12))
			} else if aua12/(math.Abs(ua11r)+abs1(ua12)) <= avb12/(math.Abs(vb11r)+abs1(vb12)) {
				csq, snq, _ = Zlartg(toCmplx(-ua11r), cmplx.Conj(ua12))
			} else {
				csq, snq, _ = Zlartg(toCmplx(-vb11r), cmplx.Conj(vb12))
			}

			csu = csl
			snu = -d1 * toCmplx(snl)
			csv = csr
			snv = -d1 * toCmplx(snr)

		} else {
			//           Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
			//           and (2,2) element of |U|**H *|A| and |V|**H *|B|.
			ua21 = -cmplx.Conj(d1) * toCmplx(snl*a1)
			ua22 = -cmplx.Conj(d1)*toCmplx(snl)*a2 + toCmplx(csl*a3)

			vb21 = -cmplx.Conj(d1) * toCmplx(snr*b1)
			vb22 = -cmplx.Conj(d1)*toCmplx(snr)*b2 + toCmplx(csr*b3)

			aua22 = math.Abs(snl)*abs1(a2) + math.Abs(csl)*math.Abs(a3)
			avb22 = math.Abs(snr)*abs1(b2) + math.Abs(csr)*math.Abs(b3)

			//           zero (2,2) elements of U**H *A and V**H *B, and then swap.
			if (abs1(ua21) + abs1(ua22)) == zero {
				csq, snq, _ = Zlartg(-cmplx.Conj(vb21), cmplx.Conj(vb22))
			} else if (abs1(vb21) + cmplx.Abs(vb22)) == zero {
				csq, snq, _ = Zlartg(-cmplx.Conj(ua21), cmplx.Conj(ua22))
			} else if aua22/(abs1(ua21)+abs1(ua22)) <= avb22/(abs1(vb21)+abs1(vb22)) {
				csq, snq, _ = Zlartg(-cmplx.Conj(ua21), cmplx.Conj(ua22))
			} else {
				csq, snq, _ = Zlartg(-cmplx.Conj(vb21), cmplx.Conj(vb22))
			}

			csu = snl
			snu = d1 * complex(csl, 0)
			csv = snr
			snv = d1 * complex(csr, 0)

		}

	} else {
		//        Input matrices A and B are lower triangular matrices
		//
		//        Form matrix C = A*adj(B) = ( a 0 )
		//                                   ( c d )
		a = a1 * b3
		d = a3 * b1
		c = a2*toCmplx(b3) - toCmplx(a3)*b2
		fc = cmplx.Abs(c)

		//        Transform complex 2-by-2 matrix C to real matrix by unitary
		//        diagonal matrix diag(d1,1).
		d1 = toCmplx(one)
		if fc != zero {
			d1 = c / complex(fc, 0)
		}

		//        The SVD of real 2 by 2 triangular C
		//
		//         ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 )
		//         ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T )
		_, _, snr, csr, snl, csl = Dlasv2(a, fc, d)

		if math.Abs(csr) >= math.Abs(snr) || math.Abs(csl) >= math.Abs(snl) {
			//           Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
			//           and (2,1) element of |U|**H *|A| and |V|**H *|B|.
			ua21 = -d1*toCmplx(snr*a1) + toCmplx(csr)*a2
			ua22r = csr * a3

			vb21 = -d1*toCmplx(snl*b1) + toCmplx(csl)*b2
			vb22r = csl * b3

			aua21 = math.Abs(snr)*math.Abs(a1) + math.Abs(csr)*abs1(a2)
			avb21 = math.Abs(snl)*math.Abs(b1) + math.Abs(csl)*abs1(b2)

			//           zero (2,1) elements of U**H *A and V**H *B.
			if (abs1(ua21) + math.Abs(ua22r)) == zero {
				csq, snq, _ = Zlartg(toCmplx(vb22r), vb21)
			} else if (abs1(vb21) + math.Abs(vb22r)) == zero {
				csq, snq, _ = Zlartg(toCmplx(ua22r), ua21)
			} else if aua21/(abs1(ua21)+math.Abs(ua22r)) <= avb21/(abs1(vb21)+math.Abs(vb22r)) {
				csq, snq, _ = Zlartg(toCmplx(ua22r), ua21)
			} else {
				csq, snq, _ = Zlartg(toCmplx(vb22r), vb21)
			}

			csu = csr
			snu = -cmplx.Conj(d1) * toCmplx(snr)
			csv = csl
			snv = -cmplx.Conj(d1) * toCmplx(snl)

		} else {
			//           Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
			//           and (1,1) element of |U|**H *|A| and |V|**H *|B|.
			ua11 = toCmplx(csr*a1) + cmplx.Conj(d1)*toCmplx(snr)*a2
			ua12 = cmplx.Conj(d1) * toCmplx(snr*a3)

			vb11 = toCmplx(csl*b1) + cmplx.Conj(d1)*toCmplx(snl)*b2
			vb12 = cmplx.Conj(d1) * toCmplx(snl*b3)

			aua11 = math.Abs(csr)*math.Abs(a1) + math.Abs(snr)*abs1(a2)
			avb11 = math.Abs(csl)*math.Abs(b1) + math.Abs(snl)*abs1(b2)

			//           zero (1,1) elements of U**H *A and V**H *B, and then swap.
			if (abs1(ua11) + abs1(ua12)) == zero {
				csq, snq, _ = Zlartg(vb12, vb11)
			} else if (abs1(vb11) + abs1(vb12)) == zero {
				csq, snq, _ = Zlartg(ua12, ua11)
			} else if aua11/(abs1(ua11)+abs1(ua12)) <= avb11/(abs1(vb11)+abs1(vb12)) {
				csq, snq, _ = Zlartg(ua12, ua11)
			} else {
				csq, snq, _ = Zlartg(vb12, vb11)
			}

			csu = snr
			snu = cmplx.Conj(d1) * toCmplx(csr)
			csv = snl
			snv = cmplx.Conj(d1) * toCmplx(csl)

		}

	}

	return
}
