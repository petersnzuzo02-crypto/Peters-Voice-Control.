package com.peters.voicecontrol

import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.*

/** Reproduces the Python training path: 16 kHz, 20 ms RMS/10 ms hop trim,
 * librosa melspectrogram (Slaney, 128 mel bands, Hann, 400 FFT, 160 hop),
 * power_to_db(top_db=80), orthonormal DCT-II, 13 MFCC coefficients,
 * 160 temporal frames then training-set coefficient normalization.
 * Cross-runtime MFCC numerical parity has NOT yet been established.
 */
object AudioFeatures {
    private const val SR = 16000
    private const val NFFT = 400
    private const val HOP = 160
    private const val MEL_BANDS = 128
    private const val COEFFS = 13
    private const val FRAMES = 160
    private val fft = FloatFFT_1D(NFFT.toLong())
    private val hann = FloatArray(NFFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / NFFT)).toFloat() }
    private val melFilters = makeFilters()

    fun trim(raw: FloatArray): FloatArray {
        if (raw.size < 320) return raw
        val rms = ArrayList<Double>()
        val starts = ArrayList<Int>()
        var s = 0
        while (s + 320 <= raw.size) {
            var power = 0.0
            for (i in s until s + 320) power += raw[i].toDouble() * raw[i]
            rms.add(sqrt(power / 320.0))
            starts.add(s)
            s += 160
        }
        val max = rms.maxOrNull() ?: return raw
        if (max <= 0.0) return raw
        val active = rms.indices.filter { rms[it] > 0.10 * max }
        if (active.isEmpty()) return raw
        val first = max(0, starts[active.first()] - 2400)
        val last = min(raw.size, starts[active.last()] + 320 + 2400)
        return raw.copyOfRange(first, last)
    }

    fun extract(raw: FloatArray, mean: FloatArray, std: FloatArray): Array<Array<FloatArray>> {
        require(mean.size == COEFFS && std.size == COEFFS)
        val signal = trim(raw)
        // librosa.center=True pads 200 samples at either end with reflect mode.
        val padded = FloatArray(signal.size + NFFT)
        for (i in padded.indices) {
            val index = reflectIndex(i - NFFT / 2, signal.size)
            padded[i] = if (signal.isNotEmpty()) signal[index] else 0f
        }
        val numFrames = 1 + (padded.size - NFFT) / HOP
        val melPowers = Array(numFrames) { DoubleArray(MEL_BANDS) }
        val fftIn = FloatArray(NFFT * 2)
        val spectrum = DoubleArray(NFFT / 2 + 1)
        for (t in 0 until numFrames) {
            java.util.Arrays.fill(fftIn, 0f)
            for (i in 0 until NFFT) fftIn[i] = padded[t * HOP + i] * hann[i]
            synchronized(fft) { fft.realForwardFull(fftIn) }
            for (k in spectrum.indices) {
                val re = fftIn[2*k].toDouble()
                val im = fftIn[2*k+1].toDouble()
                spectrum[k] = re*re + im*im
            }
            for (m in 0 until MEL_BANDS) {
                var v = 0.0
                for (k in spectrum.indices) v += melFilters[m][k] * spectrum[k]
                melPowers[t][m] = v
            }
        }
        // librosa.power_to_db: 10*log10(max(power, 1e-10)), top_db=80.
        var peakDb = Double.NEGATIVE_INFINITY
        val db = Array(numFrames) { DoubleArray(MEL_BANDS) }
        for (t in 0 until numFrames) for (m in 0 until MEL_BANDS) {
            val value = 10.0 * log10(max(1e-10, melPowers[t][m]))
            db[t][m] = value
            if (value > peakDb) peakDb = value
        }
        val floor = peakDb - 80.0
        val output = Array(1) { Array(FRAMES) { FloatArray(COEFFS) } }
        for (t in 0 until min(FRAMES, numFrames)) {
            for (c in 0 until COEFFS) {
                var sum = 0.0
                for (m in 0 until MEL_BANDS) {
                    sum += max(floor, db[t][m]) * cos(PI * c * (m + 0.5) / MEL_BANDS)
                }
                val dctScale = if (c == 0) sqrt(1.0 / MEL_BANDS) else sqrt(2.0 / MEL_BANDS)
                val mfcc = sum * dctScale
                output[0][t][c] = ((mfcc - mean[c]) / max(std[c].toDouble(), 1e-6)).toFloat()
            }
        }
        // Frames beyond the actual duration remain zero after normalization,
        // matching the training notebook's padded-frame mask.
        return output
    }

    private fun reflectIndex(index: Int, n: Int): Int {
        if (n <= 1) return 0
        val period = 2 * (n - 1)
        val z = ((index % period) + period) % period
        return if (z < n) z else period - z
    }

    private fun hzToMel(hz: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        return if (hz < minLogHz) hz / fSp
        else minLogMel + ln(hz / minLogHz) / ln(6.4) * 27.0
    }
    private fun melToHz(mel: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        return if (mel < minLogMel) mel * fSp
        else minLogHz * exp(ln(6.4) * (mel - minLogMel) / 27.0)
    }
    private fun makeFilters(): Array<DoubleArray> {
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(SR / 2.0)
        val hz = DoubleArray(MEL_BANDS + 2) { i ->
            melToHz(melMin + (melMax - melMin) * i / (MEL_BANDS + 1))
        }
        return Array(MEL_BANDS) { m ->
            val lower = hz[m]
            val center = hz[m+1]
            val upper = hz[m+2]
            val enorm = 2.0 / (upper - lower)
            DoubleArray(NFFT / 2 + 1) { k ->
                val freq = k.toDouble() * SR / NFFT
                val left = (freq - lower) / (center - lower)
                val right = (upper - freq) / (upper - center)
                max(0.0, min(left, right)) * enorm
            }
        }
    }
}
